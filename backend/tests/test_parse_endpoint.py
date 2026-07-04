# ruff: noqa: E501
"""Unit and integration tests for the /parse endpoint (Phase 7.2)."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import fitz  # PyMuPDF
import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app.main import app
from app.schemas.parsing import ParsedJobDescription, ParsedResume

client = TestClient(app, raise_server_exceptions=False)


def _make_text_pdf(path: Path, text: str) -> None:
    c = canvas.Canvas(str(path))
    for i, line in enumerate(text.splitlines() or [text]):
        c.drawString(72, 720 - i * 16, line)
    c.showPage()
    c.save()


def test_parse_resume_success(tmp_path: Path) -> None:
    """Verify that a valid PDF resume is successfully parsed and returns a conformant ParsedResume schema."""
    pdf_path = tmp_path / "valid_resume.pdf"
    resume_text = (
        "Jane Doe\n"
        "jane.doe@example.com\n"
        "SKILLS\n"
        "Python, SQL, Docker\n"
        "EXPERIENCE\n"
        "Senior Developer at Acme Corp, Jan 2019 - Present\n"
        "Built microservices using Python.\n"
        "EDUCATION\n"
        "Bachelor of Science in Computer Science, 2015\n"
    )
    _make_text_pdf(pdf_path, resume_text)

    with open(pdf_path, "rb") as f:
        response = client.post(
            "/api/v1/parse",
            files={"file": (pdf_path.name, f, "application/pdf")},
            data={"document_type": "resume"},
        )

    assert response.status_code == 200
    data = response.json()

    # Validate schema fields
    parsed = ParsedResume(**data)
    assert parsed.contact_info_present is True
    assert "Python" in parsed.skills
    assert "Acme Corp" in parsed.experience[0].company
    assert parsed.parsing_confidence > 0.0


def test_parse_jd_text_success() -> None:
    """Verify that raw JD text submitted via form fields returns a conformant ParsedJobDescription."""
    jd_text = (
        "Looking for a Senior Python Developer with 5+ years of experience.\n"
        "Must know SQL, Docker, and Kubernetes.\n"
        "Degree required: Bachelor's."
    )

    response = client.post(
        "/api/v1/parse",
        data={"jd_text": jd_text, "document_type": "jd"},
    )

    assert response.status_code == 200
    data = response.json()

    parsed = ParsedJobDescription(**data)
    assert "Python" in parsed.required_skills
    assert parsed.required_years_experience == 5.0
    assert parsed.required_education_level == "Bachelor's"
    assert parsed.parsing_confidence > 0.0


def test_parse_jd_file_success(tmp_path: Path) -> None:
    """Verify that a JD uploaded as a PDF file is successfully parsed."""
    pdf_path = tmp_path / "job_desc.pdf"
    jd_text = (
        "Required years of experience: 3+ years.\n"
        "Skills: Python, Java.\n"
        "Required education: Master's.\n"
    )
    _make_text_pdf(pdf_path, jd_text)

    with open(pdf_path, "rb") as f:
        response = client.post(
            "/api/v1/parse",
            files={"file": (pdf_path.name, f, "application/pdf")},
            data={"document_type": "jd"},
        )

    assert response.status_code == 200
    data = response.json()

    parsed = ParsedJobDescription(**data)
    assert "Python" in parsed.required_skills
    assert parsed.required_years_experience == 3.0
    assert parsed.required_education_level == "Master's"
    assert parsed.parsing_confidence > 0.0


def test_parse_file_too_large() -> None:
    """Verify that files exceeding the strict 5MB limit are rejected with HTTP 413."""
    large_payload = b"0" * (5 * 1024 * 1024 + 1)  # 5MB + 1 byte

    response = client.post(
        "/api/v1/parse",
        files={"file": ("huge.pdf", large_payload, "application/pdf")},
        data={"document_type": "resume"},
    )

    assert response.status_code == 413
    assert "File too large" in response.json()["message"]


def test_parse_invalid_signature() -> None:
    """Verify that PDF uploads are signature-sniffed and rejected if magic bytes are invalid."""
    fake_pdf = b"NOT_A_REAL_PDF_MAGIC_BYTES_12345"

    response = client.post(
        "/api/v1/parse",
        files={"file": ("fake.pdf", fake_pdf, "application/pdf")},
        data={"document_type": "resume"},
    )

    assert response.status_code == 400
    assert "Invalid file signature" in response.json()["message"]


def test_parse_empty_file() -> None:
    """Verify that empty uploads are caught and rejected immediately."""
    response = client.post(
        "/api/v1/parse",
        files={"file": ("empty.pdf", b"", "application/pdf")},
        data={"document_type": "resume"},
    )

    assert response.status_code == 400
    assert "Uploaded file is empty" in response.json()["message"]


def test_parse_garbage_pdf(tmp_path: Path) -> None:
    """Verify that a blank/image-only PDF (no extractable text) returns a 400 Bad Request per PRD §8.2."""
    pdf_path = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()

    with open(pdf_path, "rb") as f:
        response = client.post(
            "/api/v1/parse",
            files={"file": (pdf_path.name, f, "application/pdf")},
            data={"document_type": "resume"},
        )

    assert response.status_code == 400
    data = response.json()
    assert "too short" in data["message"].lower() or "no recognizable words" in data["message"].lower()


def test_parse_internal_error_hidden(tmp_path: Path) -> None:
    """Verify that unhandled pipeline errors return a safe 500 without leaking stack traces or raw details to client."""
    pdf_path = tmp_path / "valid_resume.pdf"
    resume_text = (
        "Jane Doe\njane.doe@example.com\n"
        "Experience: Python software developer with five years of experience building web applications.\n"
        "Skills: Python, SQL, Git, Docker, backend development."
    )
    _make_text_pdf(pdf_path, resume_text)

    with patch(
        "app.api.v1.endpoints.parse.structure_resume",
        side_effect=RuntimeError("Database connection lost!"),
    ):
        with open(pdf_path, "rb") as f:
            response = client.post(
                "/api/v1/parse",
                files={"file": (pdf_path.name, f, "application/pdf")},
                data={"document_type": "resume"},
            )

    assert response.status_code == 500
    data = response.json()
    assert data["code"] == "INTERNAL_SERVER_ERROR"
    # Ensure raw details and stack trace strings are NOT in response
    assert "Database connection lost" not in data["message"]
    assert "Traceback" not in data["message"]


def test_parse_pii_logging_prevention(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify that no PII (e.g. name, email, phone) from the resume contents is printed to system logs."""
    pdf_path = tmp_path / "confidential_resume.pdf"
    secret_name = "Montgomery Burns"
    secret_email = "monty.burns@springfieldnuclear.com"
    secret_phone = "+1 555 867 5309"

    resume_text = (
        f"{secret_name}\n"
        f"{secret_email} | {secret_phone}\n"
        "SKILLS\n"
        "Nuclear Reactor Operation, Python\n"
        "EXPERIENCE\n"
        "Owner at Springfield Nuclear Power Plant, Jan 1980 - Present\n"
    )
    _make_text_pdf(pdf_path, resume_text)

    with caplog.at_level(logging.INFO):
        with open(pdf_path, "rb") as f:
            response = client.post(
                "/api/v1/parse",
                files={"file": (pdf_path.name, f, "application/pdf")},
                data={"document_type": "resume"},
            )

    assert response.status_code == 200

    # Ensure no PII was written to the captured logs
    for record in caplog.records:
        log_message = record.getMessage()
        assert secret_name not in log_message, "PII Name leaked in logs!"
        assert secret_email not in log_message, "PII Email leaked in logs!"
        assert secret_phone not in log_message, "PII Phone leaked in logs!"

# ruff: noqa: E501
"""Unit and integration tests for the shared guardrails module (Phase 7.7).

Verifies validation logic for:
  - File uploads (empty, oversized, magic signatures)
  - Text input (presence, whitespace)
  - Content quality / English heuristics (PRD §8.2 gap check)
  - Batch size ceilings (latency budget validation)
  - Integration with existing FastAPI routers to prove HTTP status and error envelope.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.v1.guardrails import (
    validate_file_upload,
    validate_text_input,
    detect_content_quality,
    validate_batch_size,
    MAX_FILE_SIZE_BYTES,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def override_auth():
    """Use FastAPI dependency overrides to mock recruiter authentication."""
    from app.core.auth import get_current_recruiter, RecruiterAccount
    app.dependency_overrides[get_current_recruiter] = lambda: RecruiterAccount(
        account_id="company_a", recruiter_id="recruiter_one"
    )
    yield
    app.dependency_overrides.pop(get_current_recruiter, None)


# ============================ UNIT TESTS =====================================

def test_validate_file_upload_empty():
    res = validate_file_upload(b"")
    assert not res.is_valid
    assert res.error_code == "EMPTY_FILE"
    assert "empty" in res.error_detail.lower()


def test_validate_file_upload_oversized():
    # Simulate a file larger than MAX_FILE_SIZE_BYTES
    large_bytes = b"x" * (MAX_FILE_SIZE_BYTES + 1)
    res = validate_file_upload(large_bytes)
    assert not res.is_valid
    assert res.error_code == "FILE_TOO_LARGE"
    assert res.http_status == 413
    assert "too large" in res.error_detail.lower()


def test_validate_file_upload_valid_pdf():
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n..."
    res = validate_file_upload(pdf_bytes, filename="resume.pdf")
    assert res.is_valid


def test_validate_file_upload_invalid_pdf_signature():
    # File ends in .pdf but signature is spoofed
    bad_pdf_bytes = b"Hello, this is a plain text file spoofing as PDF."
    res = validate_file_upload(bad_pdf_bytes, filename="resume.pdf")
    assert not res.is_valid
    assert res.error_code == "INVALID_FILE_SIGNATURE"
    assert "signature" in res.error_detail.lower()


def test_validate_file_upload_valid_txt():
    # Plain text file should skip PDF signature checks
    txt_bytes = b"This is a plain text resume content."
    res = validate_file_upload(txt_bytes, filename="resume.txt")
    assert res.is_valid


def test_validate_text_input_empty():
    res1 = validate_text_input(None, "raw_resume_text")
    assert not res1.is_valid
    assert "must be provided" in res1.error_detail

    res2 = validate_text_input("   ", "raw_resume_text")
    assert not res2.is_valid
    assert "must be provided" in res2.error_detail


def test_validate_text_input_valid():
    res = validate_text_input("Software Engineer with 5 years experience.", "raw_resume_text")
    assert res.is_valid


def test_detect_content_quality_too_short():
    res = detect_content_quality("Sho")
    assert not res.is_acceptable
    assert "too short" in res.reason.lower()


def test_detect_content_quality_non_english():
    # Dominantly non-ASCII/non-Latin text (e.g. Cyrillic or Chinese characters)
    # Cyrillic: "Привет, это резюме на русском языке с некоторой информацией."
    non_english_text = "Привет, это резюме на русском языке с некоторой информацией для проверки."
    res = detect_content_quality(non_english_text)
    assert not res.is_acceptable
    assert "non-english" in res.reason.lower()


def test_detect_content_quality_garbage_words():
    # Text that is ASCII but has zero overlap with common English words
    garbage_text = "xyzabc qwertypoiu mnbvcxzlkjhg fdsaqpwoeiruty slkdfjsldkfjsldkfjs"
    res = detect_content_quality(garbage_text)
    assert not res.is_acceptable
    assert "english" in res.reason.lower() or "recognizable words" in res.reason.lower()


def test_detect_content_quality_valid_english():
    valid_text = (
        "Experienced software developer with a strong background in Python, "
        "machine learning, and web API construction. Worked on major enterprise projects "
        "responsible for structuring resume matching models and implementing clean codebase."
    )
    res = detect_content_quality(valid_text)
    assert res.is_acceptable
    assert res.detected_english_word_ratio > 0.05


def test_validate_batch_size_valid():
    res = validate_batch_size(10, 50)
    assert res.is_valid


def test_validate_batch_size_empty():
    res = validate_batch_size(0, 50)
    assert not res.is_valid
    assert res.error_code == "EMPTY_BATCH"


def test_validate_batch_size_oversized():
    res = validate_batch_size(51, 50)
    assert not res.is_valid
    assert res.error_code == "BATCH_TOO_LARGE"
    assert "exceeds" in res.error_detail


# ============================ INTEGRATION TESTS ==============================

def test_parse_endpoint_empty_file_http():
    # Send empty file
    files = {"file": ("resume.pdf", b"", "application/pdf")}
    response = client.post("/api/v1/parse", files=files, data={"document_type": "resume"})
    assert response.status_code == 400
    data = response.json()
    assert "empty" in data["message"].lower()


def test_parse_endpoint_spoofed_pdf_http():
    # Send text file with .pdf extension
    files = {"file": ("resume.pdf", b"Not a PDF file content", "application/pdf")}
    response = client.post("/api/v1/parse", files=files, data={"document_type": "resume"})
    assert response.status_code == 400
    data = response.json()
    assert "invalid file signature" in data["message"].lower()


def test_parse_endpoint_non_english_http():
    # Non-English resume text should be rejected after parsing
    # Use a dummy text file with Cyrillic characters
    non_english_content = b"\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82 \xd1\x8d\xd1\x82\xd0\xbe \xd1\x80\xd0\xb5\xd0\xb7\xd1\x8e\xd0\xbc\xd0\xb5 \xd0\xbd\xd0\xb0 \xd1\x80\xd1\x83\xd1\x81\xd1\x81\xd0\xba\xd0\xbe\xd0\xbc \xd1\x8f\xd0\xb7\xd1\x8b\xd0\xba\xd0\xb5" * 2
    files = {"file": ("resume.txt", non_english_content, "text/plain")}
    response = client.post("/api/v1/parse", files=files, data={"document_type": "resume"})
    assert response.status_code == 400
    data = response.json()
    assert "english" in data["message"].lower()


def test_score_endpoint_empty_text_http():
    # Empty inputs for score
    payload = {
        "raw_resume_text": "",
        "raw_jd_text": "Need python developer",
    }
    response = client.post("/api/v1/score", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "raw_resume_text" in data["message"]


def test_rank_endpoint_oversized_batch_http():
    # R5: batches of 51-1000 now run asynchronously; only >1000 is rejected at validation.
    resumes = [{"candidate_id": f"c-{i}", "raw_resume_text": "Python experience"} for i in range(1001)]
    payload = {
        "raw_jd_text": "Looking for python developers.",
        "resumes": resumes,
    }
    response = client.post("/api/v1/rank", json=payload)
    # Pydantic v2 max_length is now 1000 → 1001 triggers validation error (422) or our own 400 check.
    assert response.status_code in (400, 422)


def test_rank_endpoint_non_english_candidate_http():
    # One non-English candidate inside a batch should be isolated and reported as failure in responses
    from unittest.mock import patch
    from app.schemas.scoring import ScoreResult, ConfidenceLevel, FeatureVector

    mock_score = ScoreResult(
        resume_id="r-1",
        jd_id="j-1",
        final_score=85,
        feature_vector=FeatureVector(
            tfidf_score=0.8,
            embedding_score=0.8,
            skill_overlap_pct=0.8,
            exp_match=0.8,
            edu_match=0.0,
        ),
        scoring_confidence=0.8,
        confidence_level=ConfidenceLevel.HIGH,
        parsing_confidence=0.9,
        pipeline_version="v3-hybrid",
    )

    payload = {
        "raw_jd_text": "Looking for python developer with many years of experience.",
        "resumes": [
            {
                "candidate_id": "eng-1",
                "raw_resume_text": "Senior Python engineer with ten years of experience writing clean code.",
            },
            {
                "candidate_id": "non-eng-2",
                "raw_resume_text": "Привет это не английский текст который должен быть отклонен.",
            }
        ]
    }

    with patch("app.services.ranking.batch_ranking.run_orchestration", return_value=mock_score):
        response = client.post("/api/v1/rank", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_submitted"] == 2
        assert data["total_successful"] == 1
        assert data["total_failed"] == 1
        assert any("english" in f["reason"].lower() for f in data["failures"])

"""Tests for the Phase 1.1 document extraction layer.

Synthetic fixtures are generated on the fly (reportlab / PyMuPDF) since real
resumes arrive with the Kaggle dataset separately.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF, used only to fabricate an empty (zero-text) PDF fixture.
import pytest
from reportlab.pdfgen import canvas

from app.schemas.parsing import ParsingWarningCode
from app.services.extraction.pdf_extractor import (
    PDFTextExtractor,
    UnsupportedFileTypeError,
)


@pytest.fixture
def extractor() -> PDFTextExtractor:
    return PDFTextExtractor()


def _make_text_pdf(path: Path, text: str) -> None:
    c = canvas.Canvas(str(path))
    for i, line in enumerate(text.splitlines() or [text]):
        c.drawString(72, 720 - i * 16, line)
    c.showPage()
    c.save()


def test_valid_text_pdf_extracts(extractor: PDFTextExtractor, tmp_path: Path) -> None:
    pdf = tmp_path / "resume.pdf"
    _make_text_pdf(pdf, "Jane Doe\nSenior Python Engineer\nExperience at Acme Corp")

    result = extractor.extract(pdf)

    assert result.is_processable is True
    assert result.extraction_method_used in ("pdfplumber", "pymupdf")
    assert "Python Engineer" in result.raw_text
    assert ParsingWarningCode.EMPTY_DOCUMENT not in result.warnings


def test_empty_pdf_flags_and_blocks(
    extractor: PDFTextExtractor, tmp_path: Path
) -> None:
    pdf = tmp_path / "empty.pdf"
    doc = fitz.open()  # New PDF...
    doc.new_page()  # ...with a page but no text (mimics an image-only/blank scan).
    doc.save(str(pdf))
    doc.close()

    result = extractor.extract(pdf)

    assert result.is_processable is False
    # A page exists but no text ⇒ image-only and/or empty; both are hard stops.
    assert (
        ParsingWarningCode.EMPTY_DOCUMENT in result.warnings
        or ParsingWarningCode.IMAGE_ONLY_SUSPECTED in result.warnings
    )


def test_image_only_detection_flags_image_only(
    extractor: PDFTextExtractor, tmp_path: Path
) -> None:
    pdf = tmp_path / "scan.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf))
    doc.close()

    result = extractor.extract(pdf)

    assert ParsingWarningCode.IMAGE_ONLY_SUSPECTED in result.warnings
    assert result.page_count is not None and result.page_count > 0


def test_plain_text_input(extractor: PDFTextExtractor, tmp_path: Path) -> None:
    txt = tmp_path / "resume.txt"
    content = "John Smith\nData Scientist with 5 years of experience in Python."
    txt.write_text(content, encoding="utf-8")

    result = extractor.extract(txt)

    assert result.extraction_method_used == "plain_text"
    assert result.raw_text == content
    assert result.is_processable is True


def test_unsupported_file_type_raises(
    extractor: PDFTextExtractor, tmp_path: Path
) -> None:
    docx = tmp_path / "resume.docx"
    docx.write_bytes(b"PK\x03\x04 fake docx bytes")

    with pytest.raises(UnsupportedFileTypeError):
        extractor.extract(docx)


def test_garbage_detection_unit(extractor: PDFTextExtractor) -> None:
    # Mostly non-alphanumeric ⇒ garbled.
    assert extractor._is_likely_garbage("@#$%^&*()_+@#$%^&*()_+@#$%^&*") is True
    # Single repeated character ⇒ garbled.
    assert extractor._is_likely_garbage("aaaaaaaaaaaaaaaaaaaaaaaaaaaa") is True
    # Genuine content ⇒ not garbled.
    assert (
        extractor._is_likely_garbage(
            "Experienced software engineer skilled in Python and SQL."
        )
        is False
    )

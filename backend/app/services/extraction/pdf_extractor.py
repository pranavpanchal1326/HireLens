"""Document ingestion layer: raw text extraction + unreadability detection.

Scope (Phase 1.1): extract raw text from PDF/plain-text inputs and detect
empty / image-only / non-English / garbled / columnar documents. NO spaCy
structuring (Part 1.2), NO parsing_confidence (Part 1.3), NO OCR (out of scope
per PRD §11 free-tier constraint).

Design philosophy (Design Blueprint P3 / §10.10): every failure mode returns a
structured, typed ``ExtractionResult`` with machine-readable warning codes — an
unhandled exception must never reach the user as a raw 500.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
from langdetect import LangDetectException, detect

from app.schemas.parsing import ExtractionResult, ParsingWarningCode

logger = logging.getLogger(__name__)

# --- Tuning constants (named, not magic numbers) -----------------------------
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB abuse/accident guardrail (PRD §10).
MIN_CHARS_PER_PAGE = 50  # Below this avg, pdfplumber output is "suspiciously little".
EMPTY_TEXT_THRESHOLD = 10  # <= this many non-whitespace chars ≈ no text recovered.
GARBAGE_NON_ALNUM_RATIO = 0.6  # >60% non-alphanumeric ⇒ likely broken extraction.
GARBAGE_MIN_LENGTH = 20  # Don't judge garbage on trivially short strings.
NON_ENGLISH_MIN_LENGTH = 40  # langdetect is unreliable on very short text.


class UnsupportedFileTypeError(Exception):
    """Raised for inputs that are neither PDF nor plain text (PRD §3.1 scope)."""


class FileTooLargeError(Exception):
    """Raised when an input exceeds ``MAX_FILE_SIZE_BYTES`` before extraction."""


class PDFTextExtractor:
    """Extracts raw text and flags unreadable documents at the earliest point."""

    def extract(self, file_path: str | Path) -> ExtractionResult:
        """Extract raw text from a PDF or .txt file into a typed result.

        Never raises for content problems (empty/image-only/garbled) — those come
        back as an ``ExtractionResult`` with warnings. Only raises for pre-flight
        contract violations: unsupported type, missing file, oversize file.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")

        size = path.stat().st_size
        if size > MAX_FILE_SIZE_BYTES:
            raise FileTooLargeError(
                f"File {path.name} is {size} bytes; limit is {MAX_FILE_SIZE_BYTES}."
            )

        suffix = path.suffix.lower()
        if suffix == ".txt":
            return self._extract_plain_text(path)
        if suffix == ".pdf":
            return self._extract_pdf(path)
        raise UnsupportedFileTypeError(
            f"Unsupported file type {suffix!r}. Only .pdf and .txt are accepted."
        )

    # --- Plain text -----------------------------------------------------------
    def _extract_plain_text(self, path: Path) -> ExtractionResult:
        text = path.read_text(encoding="utf-8", errors="replace")
        warnings = self._content_warnings(text)
        return ExtractionResult(
            raw_text=text,
            extraction_method_used="plain_text",
            warnings=warnings,
            is_processable=self._is_processable(warnings),
            page_count=None,
        )

    # --- PDF ------------------------------------------------------------------
    def _extract_pdf(self, path: Path) -> ExtractionResult:
        """Extract PDF text.

        Fallback order — pdfplumber FIRST, PyMuPDF SECOND: pdfplumber preserves
        column/table structure better (critical for two-column resumes, PRD §8.2),
        but is more likely to yield little/no text on some encodings; PyMuPDF's
        fitz is faster and more robust as a raw-text fallback. So we prefer the
        higher-fidelity extractor and only fall back when it underperforms.
        """
        warnings: list[ParsingWarningCode] = []

        plumber_text, page_count, has_columns, plumber_ok = self._try_pdfplumber(path)
        if has_columns:
            warnings.append(ParsingWarningCode.TABLE_OR_COLUMN_LAYOUT_DETECTED)

        method: str = "pdfplumber"
        text = plumber_text
        avg_per_page = len(plumber_text.strip()) / page_count if page_count else 0.0
        needs_fallback = (not plumber_ok) or (
            page_count is not None and avg_per_page < MIN_CHARS_PER_PAGE
        )

        if needs_fallback:
            fitz_text, fitz_pages, fitz_ok = self._try_pymupdf(path)
            # Only adopt the fallback if it genuinely did better.
            if fitz_ok and len(fitz_text.strip()) > len(plumber_text.strip()):
                text = fitz_text
                method = "pymupdf"
                page_count = fitz_pages if fitz_pages is not None else page_count
                warnings.append(ParsingWarningCode.EXTRACTION_FALLBACK_USED)
            elif not plumber_ok and not fitz_ok:
                # Both extractors genuinely failed (corrupted file, not image-only).
                logger.warning("Both extractors failed for %s", path.name)

        warnings.extend(self._content_warnings(text))
        # Image-only heuristic needs the page count, so evaluate it here.
        if self._is_likely_image_only(page_count, text):
            warnings.append(ParsingWarningCode.IMAGE_ONLY_SUSPECTED)

        # De-duplicate while preserving order.
        warnings = list(dict.fromkeys(warnings))

        return ExtractionResult(
            raw_text=text,
            extraction_method_used=method,  # type: ignore[arg-type]
            warnings=warnings,
            is_processable=self._is_processable(warnings),
            page_count=page_count,
        )

    def _try_pdfplumber(self, path: Path) -> tuple[str, int | None, bool, bool]:
        """Return (text, page_count, has_columns, ok)."""
        try:
            parts: list[str] = []
            has_columns = False
            with pdfplumber.open(str(path)) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    parts.append(page_text)
                    if page.find_tables():
                        has_columns = True
            return "\n".join(parts), page_count, has_columns, True
        except Exception as exc:  # noqa: BLE001 — must not crash the request.
            logger.warning("pdfplumber failed for %s: %s", path.name, exc)
            return "", None, False, False

    def _try_pymupdf(self, path: Path) -> tuple[str, int | None, bool]:
        """Return (text, page_count, ok)."""
        try:
            parts: list[str] = []
            with fitz.open(str(path)) as doc:
                page_count = doc.page_count
                for page in doc:
                    parts.append(page.get_text())
            return "\n".join(parts), page_count, True
        except Exception as exc:  # noqa: BLE001 — must not crash the request.
            logger.warning("PyMuPDF failed for %s: %s", path.name, exc)
            return "", None, False

    # --- Detection heuristics (each independently testable) --------------------
    def _content_warnings(self, text: str) -> list[ParsingWarningCode]:
        """Run the text-only heuristics and collect their warning codes."""
        warnings: list[ParsingWarningCode] = []
        if self._is_empty(text):
            warnings.append(ParsingWarningCode.EMPTY_DOCUMENT)
            return warnings  # Nothing else is meaningful on empty text.
        if self._is_likely_garbage(text):
            warnings.append(ParsingWarningCode.GARBLED_TEXT_SUSPECTED)
        if self._is_likely_non_english(text):
            warnings.append(ParsingWarningCode.NON_ENGLISH_SUSPECTED)
        return warnings

    def _is_empty(self, text: str) -> bool:
        """True when essentially no text was recovered."""
        return len(text.strip()) <= EMPTY_TEXT_THRESHOLD

    def _is_likely_image_only(
        self, page_count: int | None, extracted_text: str
    ) -> bool:
        """True when a PDF has pages but yielded (near-)empty text — likely scanned.

        No OCR is attempted (out of scope); the document is flagged and rejected so
        the user can be asked for a text-based PDF instead.
        """
        return bool(page_count and page_count > 0 and self._is_empty(extracted_text))

    def _is_likely_non_english(self, text: str) -> bool:
        """Lightweight non-English heuristic via langdetect (no ML model of ours)."""
        sample = text.strip()
        if len(sample) < NON_ENGLISH_MIN_LENGTH:
            return False
        try:
            return bool(detect(sample) != "en")
        except LangDetectException:
            return False

    def _is_likely_garbage(self, text: str) -> bool:
        """Heuristic for corrupted/garbled extraction.

        Flags text that is mostly non-alphanumeric or a single repeated character —
        both signatures of a broken extraction rather than genuine content.
        """
        stripped = text.strip()
        if len(stripped) < GARBAGE_MIN_LENGTH:
            return False
        non_space = re.sub(r"\s", "", stripped)
        if not non_space:
            return False
        non_alnum = sum(1 for ch in non_space if not ch.isalnum())
        if non_alnum / len(non_space) > GARBAGE_NON_ALNUM_RATIO:
            return True
        # Single character (ignoring whitespace) repeated throughout.
        if len(set(non_space)) == 1:
            return True
        return False

    # --- Processability -------------------------------------------------------
    def _is_processable(self, warnings: list[ParsingWarningCode]) -> bool:
        """False only for hard-stop conditions; soft warnings continue processing."""
        hard_stops = {
            ParsingWarningCode.EMPTY_DOCUMENT,
            ParsingWarningCode.IMAGE_ONLY_SUSPECTED,
        }
        return not any(w in hard_stops for w in warnings)

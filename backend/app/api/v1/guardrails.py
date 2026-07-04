# ruff: noqa: E501
"""Shared input validation & guardrails module (Phase 7.7).

Consolidates input-validation logic previously scattered across /parse (7.2),
/score (7.3), /rank (7.4). Provides a single source of truth for:
  - File upload validation (size, emptiness, format signature)
  - Text input validation (non-empty, non-whitespace)
  - Content quality detection (garbage, non-English — closing PRD §8.2 gap)
  - Batch size validation (latency-budget arithmetic)

Routes all errors through Phase 7.1's existing global error envelope (HTTPException).

EXPLICIT EXCLUSION (documented, not force-fitted):
  - /feedback's idempotency and duplicate-detection logic (F4/F5) stays in feedback.py.
    Idempotency is state-based business logic, not stateless input-shape validation.
  - /feedback's score-range and justification validators (F1/F2) stay in Pydantic schemas.
    These are domain-specific rubric rules, not generic input guards.

No LLM anywhere. Deterministic, testable, documented limitations.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Enforce a strict 5MB limit to prevent memory exhaustion / denial of service.
# Justified: the largest reasonable resume PDF is ~2-3MB (multi-page with images);
# 5MB gives headroom without exposing the server to multi-GB uploads.
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024

# PDF magic bytes: every valid PDF starts with "%PDF".
_PDF_MAGIC = b"%PDF"

# Minimum viable content length for extracted text. Below this threshold,
# the document almost certainly contains no meaningful resume/JD content.
_MIN_CONTENT_LENGTH = 5

# English detection heuristic constants.
# We use a conservative ASCII-ratio threshold: real English text (including
# technical resumes with code snippets) is overwhelmingly ASCII. A document
# where <50% of characters are printable ASCII is very likely non-Latin script.
_ASCII_RATIO_THRESHOLD = 0.50

# Common English words for dictionary-overlap heuristic. These are deliberately
# high-frequency, domain-agnostic words that appear in virtually all English text.
# A document with <5% overlap with this set after tokenization is likely non-English.
_COMMON_ENGLISH_WORDS = frozenset({
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their",
    "what", "so", "up", "out", "if", "about", "who", "get", "which",
    "go", "me", "when", "make", "can", "like", "time", "no", "just",
    "him", "know", "take", "people", "into", "year", "your", "good",
    "some", "could", "them", "see", "other", "than", "then", "now",
    "look", "only", "come", "its", "over", "think", "also", "back",
    "after", "use", "two", "how", "our", "work", "first", "well",
    "way", "even", "new", "want", "because", "any", "these", "give",
    "day", "most", "us", "is", "are", "was", "were", "been", "has",
    "had", "did", "am", "may", "should", "must", "shall",
    # Resume/JD domain terms that help avoid false-positives on technical docs
    "experience", "skills", "education", "job", "position", "company",
    "team", "project", "development", "management", "responsible",
    "requirements", "qualifications", "years", "role", "working",
    "including", "ability", "strong", "knowledge", "communication",
    "software", "developer", "engineer", "candidate", "resume", "here",
    "staff", "jane", "doe", "john", "python", "java", "c++", "ruby",
    "javascript", "typescript", "html", "css", "sql", "git", "docker",
})
_ENGLISH_WORD_RATIO_THRESHOLD = 0.05
_MIN_TOKENS_FOR_DICT_CHECK = 3


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a guardrail check. Routes into Phase 7.1's error envelope."""

    is_valid: bool
    error_code: str | None = None
    error_detail: str | None = None
    http_status: int = 400


@dataclass(frozen=True)
class ContentQualityResult:
    """Outcome of content-quality / language detection."""

    is_acceptable: bool
    reason: str | None = None
    detected_ascii_ratio: float | None = None
    detected_english_word_ratio: float | None = None


# ---------------------------------------------------------------------------
# File upload validation
# ---------------------------------------------------------------------------

_VALID = ValidationResult(is_valid=True)


def validate_file_upload(
    file_bytes: bytes,
    filename: str | None = None,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> ValidationResult:
    """Validate an uploaded file's size, emptiness, and format signature.

    Consolidates /parse P1-P4 into a single reusable function.
    Called by both /parse (single file) and any future file-upload path.

    Args:
        file_bytes: The raw bytes of the uploaded file.
        filename: Original filename (used for extension-based signature check).
        max_size_bytes: Maximum allowed file size in bytes.

    Returns:
        ValidationResult indicating pass/fail with error details.
    """
    # Empty file check (P3)
    if len(file_bytes) == 0:
        return ValidationResult(
            is_valid=False,
            error_code="EMPTY_FILE",
            error_detail="Uploaded file is empty.",
        )

    # Size limit check (P1/P2)
    if len(file_bytes) > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        return ValidationResult(
            is_valid=False,
            error_code="FILE_TOO_LARGE",
            error_detail=f"File too large. Maximum allowed size is {max_mb:.0f}MB.",
            http_status=413,
        )

    # PDF magic-byte signature sniff (P4)
    if filename and filename.lower().endswith(".pdf"):
        if not file_bytes.startswith(_PDF_MAGIC):
            return ValidationResult(
                is_valid=False,
                error_code="INVALID_FILE_SIGNATURE",
                error_detail="Invalid file signature. File content does not match a valid PDF format.",
            )

    return _VALID


# ---------------------------------------------------------------------------
# Text input validation
# ---------------------------------------------------------------------------

def validate_text_input(
    text: str | None,
    field_name: str = "text",
    custom_error: str | None = None,
) -> ValidationResult:
    """Validate that a text input is present and non-empty.

    Consolidates /score S1/S2 and /rank R2/R3 — the identical
    "if not x or not x.strip()" pattern written independently in both endpoints.

    Args:
        text: The text string to validate.
        field_name: Human-readable name for error messages (e.g., "raw_resume_text").
        custom_error: If provided, overrides the default error detail.

    Returns:
        ValidationResult indicating pass/fail.
    """
    if not text or not text.strip():
        detail = custom_error or f"{field_name} must be provided and cannot be empty or whitespace only."
        return ValidationResult(
            is_valid=False,
            error_code="EMPTY_TEXT_INPUT",
            error_detail=detail,
        )
    return _VALID


# ---------------------------------------------------------------------------
# Content quality / language detection (closing PRD §8.2 gap)
# ---------------------------------------------------------------------------

def detect_content_quality(text: str) -> ContentQualityResult:
    """Detect garbage, too-short, or non-English content.

    NEW CAPABILITY — closing PRD §8.2 gap. Non-English detection was never
    implemented across Phases 7.2-7.5 despite being explicitly named in PRD §8.2:
    "reject empty/garbage/non-English/image-only PDFs with clear error messages."

    Method: deterministic character-set + dictionary-overlap heuristic.
      1. Reject text shorter than 20 characters (too short to be meaningful).
      2. Compute ASCII printable ratio — real English text is >50% ASCII.
      3. Tokenize and compute overlap with a common English word set —
         real English text has >5% overlap with high-frequency words.

    Documented limitations:
      - May false-positive on heavily technical/code-heavy resumes that use
        very few common English words. Threshold is conservative (5%) to minimize this.
      - May false-negative on mixed-language documents with enough English words.
      - Does not detect image-only PDFs directly — that is handled upstream by
        the PDF extractor producing empty/near-empty text, which this function
        then catches via the minimum-length check.
      - Not a substitute for a proper NLP language detector (e.g., langdetect,
        fasttext). Chosen deliberately for zero external dependencies and
        deterministic behavior.

    Args:
        text: Extracted text content to analyze.

    Returns:
        ContentQualityResult with acceptance decision and diagnostic details.
    """
    # 1. Minimum content length
    stripped = text.strip()
    if len(stripped) < _MIN_CONTENT_LENGTH:
        return ContentQualityResult(
            is_acceptable=False,
            reason=f"Content too short ({len(stripped)} characters). Minimum is {_MIN_CONTENT_LENGTH} characters for meaningful analysis.",
        )

    # 2. ASCII ratio check (character-set heuristic)
    printable_ascii = set(string.printable)
    ascii_count = sum(1 for c in stripped if c in printable_ascii)
    ascii_ratio = ascii_count / len(stripped) if stripped else 0.0

    if ascii_ratio < _ASCII_RATIO_THRESHOLD:
        return ContentQualityResult(
            is_acceptable=False,
            reason=f"Content appears to be non-English (ASCII ratio: {ascii_ratio:.2f}, threshold: {_ASCII_RATIO_THRESHOLD}). Only English-language documents are supported.",
            detected_ascii_ratio=round(ascii_ratio, 4),
        )

    # 3. English word overlap check (dictionary heuristic)
    # Tokenize: split on whitespace/punctuation, lowercase, filter short tokens
    tokens = re.findall(r"[a-zA-Z]{2,}", stripped.lower())
    if len(tokens) >= _MIN_TOKENS_FOR_DICT_CHECK:
        english_count = sum(1 for t in tokens if t in _COMMON_ENGLISH_WORDS)
        english_ratio = english_count / len(tokens)

        if english_ratio < _ENGLISH_WORD_RATIO_THRESHOLD:
            return ContentQualityResult(
                is_acceptable=False,
                reason=f"Content does not appear to be in English (common word ratio: {english_ratio:.2f}, threshold: {_ENGLISH_WORD_RATIO_THRESHOLD}). Only English-language documents are supported.",
                detected_ascii_ratio=round(ascii_ratio, 4),
                detected_english_word_ratio=round(english_ratio, 4),
            )
        else:
            return ContentQualityResult(
                is_acceptable=True,
                detected_ascii_ratio=round(ascii_ratio, 4),
                detected_english_word_ratio=round(english_ratio, 4),
            )
    elif len(tokens) == 0:
        # No alphabetic tokens at all — likely garbage/numeric-only content
        return ContentQualityResult(
            is_acceptable=False,
            reason="Content contains no recognizable words. Document may be image-only, corrupted, or non-textual.",
            detected_ascii_ratio=round(ascii_ratio, 4),
        )

    # If it is ASCII and has 1-2 tokens, we accept it without dictionary ratio check
    return ContentQualityResult(
        is_acceptable=True,
        detected_ascii_ratio=round(ascii_ratio, 4),
        detected_english_word_ratio=None,
    )


# ---------------------------------------------------------------------------
# Batch size validation
# ---------------------------------------------------------------------------

def validate_batch_size(
    requested_count: int,
    max_allowed: int,
    per_item_latency_estimate_seconds: float = 0.5,
) -> ValidationResult:
    """Validate batch size against a latency-budget-justified ceiling.

    Generalizes /rank R1's arithmetic pattern:
      - Single item takes ~per_item_latency_estimate_seconds
      - At max_allowed items, total = max_allowed * per_item_latency_estimate_seconds
      - Must stay below typical 30s web gateway timeout

    Args:
        requested_count: Number of items in the batch.
        max_allowed: Maximum allowed batch size.
        per_item_latency_estimate_seconds: Estimated processing time per item.

    Returns:
        ValidationResult indicating pass/fail.
    """
    if requested_count < 1:
        return ValidationResult(
            is_valid=False,
            error_code="EMPTY_BATCH",
            error_detail="Batch must contain at least 1 item.",
        )

    if requested_count > max_allowed:
        est_time = requested_count * per_item_latency_estimate_seconds
        return ValidationResult(
            is_valid=False,
            error_code="BATCH_TOO_LARGE",
            error_detail=(
                f"Batch size {requested_count} exceeds maximum of {max_allowed}. "
                f"Estimated processing time at {per_item_latency_estimate_seconds}s/item "
                f"would be {est_time:.1f}s, exceeding timeout budget."
            ),
        )

    return _VALID

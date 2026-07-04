"""Parsing data contracts.

Canonical shapes for parsed resumes and job descriptions.

Maps to: PRD §4 (architecture — Parser output), PRD §9 (privacy: no raw PII
stored), and the two-confidence rule from Design Blueprint §6.3 / §10.9
(``parsing_confidence`` is kept strictly distinct from ``scoring_confidence``).

Produced by: Phase 1 (Parser). Consumed by: Phase 2+ (scorer, RAG matcher,
orchestrator), Phase 8 (frontend).
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ParsingWarningCode(str, Enum):
    """Machine-readable warning codes emitted by the extraction layer (Part 1.1).

    Maps to: PRD §8.2 (input validation / guardrails) and Design Blueprint §10.10
    (warm, blameless error copy). These are MACHINE codes only; a later UI phase
    maps each to human-facing warm language without re-deriving detection logic.

    EXACTLY these values for the extraction layer's scope — do not add more here.
    """

    EMPTY_DOCUMENT = "empty_document"
    IMAGE_ONLY_SUSPECTED = "image_only_suspected"
    NON_ENGLISH_SUSPECTED = "non_english_suspected"
    GARBLED_TEXT_SUSPECTED = "garbled_text_suspected"
    # Informational (NOT failures):
    TABLE_OR_COLUMN_LAYOUT_DETECTED = "table_or_column_layout_detected"
    EXTRACTION_FALLBACK_USED = "extraction_fallback_used"

    # --- Structuring layer (Phase 1.2) ---
    SECTION_HEADERS_NOT_DETECTED = "section_headers_not_detected"
    SKILL_SECTION_EMPTY_AFTER_EXTRACTION = "skill_section_empty_after_extraction"
    EXPERIENCE_DATES_AMBIGUOUS = "experience_dates_ambiguous"
    NO_EXPERIENCE_SECTION_FOUND = "no_experience_section_found"


class ExtractionResult(BaseModel):
    """Raw-extraction output from the document ingestion layer (Part 1.1).

    Maps to: PRD §8.2. This feeds the raw_text + early parsing_warnings of
    ``ParsedResume`` downstream. ``is_processable`` is False only for hard-stop
    conditions (EMPTY_DOCUMENT, IMAGE_ONLY_SUSPECTED); other warnings are soft and
    carried forward while processing continues.
    """

    raw_text: str
    extraction_method_used: Literal["pdfplumber", "pymupdf", "plain_text"]
    warnings: list[ParsingWarningCode] = Field(default_factory=list)
    is_processable: bool
    page_count: int | None = None


class ParsedSection(BaseModel):
    """A single extracted document section (skills, experience, education, contact).

    Maps to: PRD §4 parser sub-outputs. Used as a reusable sub-object describing
    whether a given section was recovered from the raw document.
    """

    raw_text: str
    extracted_successfully: bool


class ExperienceEntry(BaseModel):
    """One work-experience record extracted from a resume.

    Maps to: PRD §4 (parser) and the experience-matcher input in the orchestrator.
    Dates are ISO 8601 partial-date strings (e.g. ``"2021-03"``); ``end_date`` of
    ``None`` denotes "present".
    """

    title: str | None = None
    company: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    description: str
    years_calculated: float | None = None


class EducationEntry(BaseModel):
    """One education record extracted from a resume.

    Maps to: PRD §4 (parser) and the education-matcher (``edu_match`` feature).
    """

    degree: str | None = None
    institution: str | None = None
    field_of_study: str | None = None
    graduation_year: int | None = None


class ParsedResume(BaseModel):
    """Canonical parsed-resume contract.

    Maps to: PRD §4 (Parser output), PRD §9 (privacy — no raw PII fields; contact
    presence tracked only via ``contact_info_present``), and PRD §8.2 (pipeline
    versioning). ``parsing_confidence`` is DISTINCT from ``scoring_confidence`` and
    must never be merged with it (Design Blueprint §6.3 / §10.9).
    """

    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    raw_text: str
    skills: list[str] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    total_years_experience: float | None = None
    contact_info_present: bool
    parsing_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of expected fields successfully extracted.",
    )
    parsing_warnings: list[str] = Field(default_factory=list)
    pipeline_version: str


class ParsedJobDescription(BaseModel):
    """Canonical parsed job-description contract.

    Maps to: PRD §4 (Parser output, JD side) and PRD §8.2 (pipeline versioning).
    """

    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    raw_text: str
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    required_years_experience: float | None = None
    required_education_level: str | None = None
    parsing_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of expected fields successfully extracted.",
    )
    pipeline_version: str

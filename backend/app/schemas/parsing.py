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

from pydantic import BaseModel, Field


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

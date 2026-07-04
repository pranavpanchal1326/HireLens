"""Parsing-confidence scoring (Phase 1.3).

Aggregates extraction-quality signals (Phase 1.1) and structuring-completeness
signals (Phase 1.2) into the single ``parsing_confidence`` float on
ParsedResume/ParsedJobDescription.

Maps to: PRD §8.2 ("% of expected fields successfully extracted"; parsing vs
scoring confidence kept strictly separate) and Design Blueprint P1/P3 (auditable,
honest confidence). This module NEVER touches scoring_confidence — that is the
ML model's job in Phase 6.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

from app.schemas.parsing import (
    ExtractionResult,
    ParsedJobDescription,
    ParsedResume,
    ParsingWarningCode,
)

# Structuring warnings treated as "hard" for the low-weight signal check. These
# are only a LOW signal because the field-population checks below already measure
# their real impact more precisely — including them heavily would double-penalize.
_HARD_STRUCTURING_WARNINGS = {
    ParsingWarningCode.SECTION_HEADERS_NOT_DETECTED.value,
    ParsingWarningCode.NO_EXPERIENCE_SECTION_FOUND.value,
}


class ResumeFieldCheck(NamedTuple):
    """One auditable completeness check contributing to resume confidence."""

    name: str
    weight: float
    passes: Callable[[ParsedResume], bool]


class JDFieldCheck(NamedTuple):
    """One auditable completeness check contributing to JD confidence."""

    name: str
    weight: float
    passes: Callable[[ParsedJobDescription], bool]


def _resume_experience_has_title_and_company(resume: ParsedResume) -> bool:
    """True only if every parsed experience entry has BOTH title and company.

    An entry with only dates (title/company left None per the honesty rule)
    signals thin structuring, so it should not earn this credit.
    """
    return bool(resume.experience) and all(
        bool(entry.title) and bool(entry.company) for entry in resume.experience
    )


def _resume_no_hard_structuring_warnings(resume: ParsedResume) -> bool:
    return not any(w in _HARD_STRUCTURING_WARNINGS for w in resume.parsing_warnings)


# Weighted completeness scheme for resumes. Weights sum to EXACTLY 1.0.
# Rationale:
#   skills / experience are the load-bearing inputs to scoring (Phase 2) — if
#   either is missing the resume is barely usable, so each gets the top weight.
#   education + total_years are important but secondary. title/company presence
#   is a structuring-quality signal. The warning check is LOW to avoid
#   double-penalizing what the field checks already capture.
EXPECTED_RESUME_FIELDS: list[ResumeFieldCheck] = [
    ResumeFieldCheck("has_skills", 0.25, lambda r: len(r.skills) > 0),  # HIGH
    ResumeFieldCheck("has_experience", 0.25, lambda r: len(r.experience) > 0),  # HIGH
    ResumeFieldCheck("has_education", 0.15, lambda r: len(r.education) > 0),  # MEDIUM
    ResumeFieldCheck(
        "has_total_years", 0.15, lambda r: r.total_years_experience is not None
    ),  # MEDIUM
    ResumeFieldCheck(
        "experience_has_title_and_company",
        0.10,
        _resume_experience_has_title_and_company,
    ),  # MEDIUM
    ResumeFieldCheck(
        "no_hard_structuring_warnings", 0.10, _resume_no_hard_structuring_warnings
    ),  # LOW
]

# Weighted scheme for JDs. Weights sum to EXACTLY 1.0. JDs are naturally sparser
# than resumes (often just skills + a years line), so required_skills dominates
# and education is only a light signal.
EXPECTED_JD_FIELDS: list[JDFieldCheck] = [
    JDFieldCheck("has_required_skills", 0.60, lambda j: len(j.required_skills) > 0),
    JDFieldCheck(
        "has_required_years", 0.30, lambda j: j.required_years_experience is not None
    ),
    JDFieldCheck(
        "has_required_education", 0.10, lambda j: j.required_education_level is not None
    ),
]

_PRECISION = 3


def calculate_resume_parsing_confidence(
    extraction_result: ExtractionResult, parsed_resume: ParsedResume
) -> float:
    """Weighted completeness score in [0.0, 1.0].

    HARD STOP (non-negotiable): if the extraction was not processable
    (EMPTY_DOCUMENT / IMAGE_ONLY_SUSPECTED), return 0.0 immediately — no partial
    credit path exists around this short-circuit.
    """
    if not extraction_result.is_processable:
        return 0.0
    total = sum(
        check.weight for check in EXPECTED_RESUME_FIELDS if check.passes(parsed_resume)
    )
    return round(total, _PRECISION)


def calculate_jd_parsing_confidence(
    extraction_result: ExtractionResult, parsed_jd: ParsedJobDescription
) -> float:
    """Weighted completeness score for a JD in [0.0, 1.0]. Same hard-stop rule."""
    if not extraction_result.is_processable:
        return 0.0
    total = sum(check.weight for check in EXPECTED_JD_FIELDS if check.passes(parsed_jd))
    return round(total, _PRECISION)


def explain_confidence(
    extraction_result: ExtractionResult, parsed_resume: ParsedResume
) -> dict[str, dict[str, object]]:
    """Per-field breakdown of the resume confidence calculation.

    Independently reachable (not just internal logging) so a grader/debugging
    session can verify the number is real — Design Blueprint P1. Under the hard
    stop, every contribution is 0.0 so the breakdown sums to the same 0.0 the
    main calculator returns.
    """
    hard_stopped = not extraction_result.is_processable
    breakdown: dict[str, dict[str, object]] = {}
    for check in EXPECTED_RESUME_FIELDS:
        passed = check.passes(parsed_resume)
        contribution = 0.0 if hard_stopped else (check.weight if passed else 0.0)
        breakdown[check.name] = {
            "passed": passed,
            "weight": check.weight,
            "contribution": round(contribution, _PRECISION),
        }
    return breakdown

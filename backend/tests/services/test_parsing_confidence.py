"""Tests for Phase 1.3 parsing-confidence scoring."""

from __future__ import annotations

from app.schemas.parsing import (
    EducationEntry,
    ExperienceEntry,
    ExtractionResult,
    ParsedResume,
)
from app.schemas.scoring import ConfidenceLevel
from app.services.confidence.confidence_utils import (
    HIGH_THRESHOLD,
    MEDIUM_THRESHOLD,
    confidence_to_band,
)
from app.services.confidence.parsing_confidence import (
    EXPECTED_RESUME_FIELDS,
    calculate_resume_parsing_confidence,
    explain_confidence,
)


def _ok_extraction() -> ExtractionResult:
    return ExtractionResult(
        raw_text="x",
        extraction_method_used="plain_text",
        warnings=[],
        is_processable=True,
        page_count=None,
    )


def _failed_extraction() -> ExtractionResult:
    return ExtractionResult(
        raw_text="",
        extraction_method_used="pdfplumber",
        warnings=[],
        is_processable=False,
        page_count=1,
    )


def _perfect_resume() -> ParsedResume:
    return ParsedResume(
        raw_text="x",
        skills=["Python", "SQL"],
        experience=[
            ExperienceEntry(
                title="Engineer",
                company="Acme",
                start_date="2020-01",
                end_date=None,
                description="work",
                years_calculated=4.0,
            )
        ],
        education=[EducationEntry(degree="Bachelor's", graduation_year=2015)],
        total_years_experience=4.0,
        contact_info_present=True,
        parsing_confidence=0.0,
        parsing_warnings=[],
        pipeline_version="parser-v1",
    )


def test_weights_sum_to_one() -> None:
    assert round(sum(c.weight for c in EXPECTED_RESUME_FIELDS), 6) == 1.0


def test_perfect_resume_scores_one() -> None:
    conf = calculate_resume_parsing_confidence(_ok_extraction(), _perfect_resume())
    assert conf == 1.0


def test_hard_stop_forces_zero_even_with_full_resume() -> None:
    # Fully-populated resume, but extraction failed → hard stop must win.
    conf = calculate_resume_parsing_confidence(_failed_extraction(), _perfect_resume())
    assert conf == 0.0


def test_missing_education_loses_exactly_medium_weight() -> None:
    resume = _perfect_resume().model_copy(update={"education": []})
    conf = calculate_resume_parsing_confidence(_ok_extraction(), resume)
    assert conf == 0.85  # 1.0 - 0.15 (has_education weight)


def test_missing_skills_and_experience_is_low_band() -> None:
    resume = _perfect_resume().model_copy(
        update={"skills": [], "experience": [], "total_years_experience": None}
    )
    conf = calculate_resume_parsing_confidence(_ok_extraction(), resume)
    # Only has_education (0.15) + no_hard_warnings (0.10) survive.
    assert conf == 0.25
    assert conf < MEDIUM_THRESHOLD
    assert confidence_to_band(conf) is ConfidenceLevel.LOW


def test_confidence_to_band_boundaries() -> None:
    assert confidence_to_band(HIGH_THRESHOLD) is ConfidenceLevel.HIGH  # exactly 0.80
    assert confidence_to_band(0.799) is ConfidenceLevel.MEDIUM
    assert (
        confidence_to_band(MEDIUM_THRESHOLD) is ConfidenceLevel.MEDIUM
    )  # exactly 0.50
    assert confidence_to_band(0.499) is ConfidenceLevel.LOW
    assert confidence_to_band(1.0) is ConfidenceLevel.HIGH
    assert confidence_to_band(0.0) is ConfidenceLevel.LOW


def test_explain_contributions_sum_matches_calculation() -> None:
    extraction, resume = _ok_extraction(), _perfect_resume().model_copy(
        update={"education": []}
    )
    breakdown = explain_confidence(extraction, resume)
    summed = round(sum(f["contribution"] for f in breakdown.values()), 3)  # type: ignore[misc]
    direct = calculate_resume_parsing_confidence(extraction, resume)
    assert summed == direct


def test_explain_respects_hard_stop() -> None:
    breakdown = explain_confidence(_failed_extraction(), _perfect_resume())
    summed = round(sum(f["contribution"] for f in breakdown.values()), 3)  # type: ignore[misc]
    assert summed == 0.0


def test_explain_structure_shape() -> None:
    breakdown = explain_confidence(_ok_extraction(), _perfect_resume())
    assert "has_skills" in breakdown
    entry = breakdown["has_skills"]
    assert set(entry.keys()) == {"passed", "weight", "contribution"}

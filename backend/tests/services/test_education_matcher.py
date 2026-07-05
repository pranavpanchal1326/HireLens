# ruff: noqa: E501
"""Tests for the R6 education matcher (edu_match) and versioned weights config."""

from __future__ import annotations

from app.schemas.parsing import EducationEntry, ParsedJobDescription, ParsedResume
from app.services.orchestration.agent_orchestrator import load_ensemble_weights
from app.services.scoring.education_matcher import EducationMatcher, degree_to_level


def _resume(degrees: list[str]) -> ParsedResume:
    return ParsedResume(
        raw_text="x",
        skills=["python"],
        education=[EducationEntry(degree=d) for d in degrees],
        contact_info_present=False,
        parsing_confidence=0.9,
        pipeline_version="v3-hybrid",
    )


def _jd(required: str | None) -> ParsedJobDescription:
    return ParsedJobDescription(
        raw_text="x",
        required_education_level=required,
        parsing_confidence=0.9,
        pipeline_version="v3-hybrid",
    )


M = EducationMatcher()


def test_degree_level_parsing() -> None:
    assert degree_to_level("PhD in CS") == 5
    assert degree_to_level("Master of Science") == 4
    assert degree_to_level("Bachelor of Arts") == 3
    assert degree_to_level("Associate Degree") == 2
    assert degree_to_level("High School Diploma") in (1, 2)  # 'diploma' or 'high school'
    assert degree_to_level(None) == 0
    assert degree_to_level("Certified Scrum Master-ish gibberish") in (0, 4)


def test_no_requirement_returns_full() -> None:
    # No JD education requirement → nothing to satisfy → 1.0.
    assert M.match(_resume(["Bachelor"]), _jd(None)) == 1.0


def test_meets_or_exceeds_scores_full() -> None:
    assert M.match(_resume(["Master of Science"]), _jd("Bachelor")) == 1.0
    assert M.match(_resume(["Bachelor"]), _jd("Bachelor")) == 1.0


def test_under_qualified_is_proportional_not_zero_gate() -> None:
    # Bachelor(3) vs required Doctorate(5) → 3/5 = 0.6, not a hard 0.
    assert M.match(_resume(["Bachelor of Science"]), _jd("PhD")) == 0.6


def test_no_recognizable_degree_scores_low() -> None:
    assert M.match(_resume([]), _jd("Master")) == 0.0


def test_highest_degree_is_used() -> None:
    # Resume with both Bachelor and Master → the Master counts.
    assert M.match(_resume(["Bachelor", "Master"]), _jd("Master")) == 1.0


def test_institution_is_ignored_bias_safety() -> None:
    # Two resumes, same degree level, different (prestige-signaling) institutions
    # must score identically — institution must NOT influence edu_match.
    r_prestige = ParsedResume(
        raw_text="x", skills=["python"],
        education=[EducationEntry(degree="Bachelor", institution="MIT")],
        contact_info_present=False, parsing_confidence=0.9, pipeline_version="v3-hybrid",
    )
    r_other = ParsedResume(
        raw_text="x", skills=["python"],
        education=[EducationEntry(degree="Bachelor", institution="Community College")],
        contact_info_present=False, parsing_confidence=0.9, pipeline_version="v3-hybrid",
    )
    jd = _jd("Bachelor")
    assert M.match(r_prestige, jd) == M.match(r_other, jd)


def test_weights_load_from_config() -> None:
    w = load_ensemble_weights("v3_hybrid")
    assert set(w) == {"tfidf_score", "embedding_score", "skill_overlap_pct", "exp_match"}
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_weights_missing_version_falls_back() -> None:
    w = load_ensemble_weights("does_not_exist")
    # Fallback constants still sum to 1.0 and have the 4 expected keys.
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert "tfidf_score" in w

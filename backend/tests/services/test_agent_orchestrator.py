"""Minimal tests for the Phase 4.1 orchestrator skeleton."""

from __future__ import annotations

import pytest

from app.schemas.parsing import ParsedJobDescription, ParsedResume
from app.schemas.scoring import ConfidenceLevel, ScoreResult
from app.services.orchestration.agent_orchestrator import (
    OrchestrationResult,
    run_orchestration,
)


def _resume(skills: list[str]) -> ParsedResume:
    return ParsedResume(
        raw_text="x",
        skills=skills,
        experience=[],
        education=[],
        total_years_experience=None,
        contact_info_present=False,
        parsing_confidence=0.7,
        parsing_warnings=[],
        pipeline_version="parser-v1",
    )


def _jd() -> ParsedJobDescription:
    return ParsedJobDescription(
        raw_text="x",
        required_skills=["Python"],
        preferred_skills=[],
        required_years_experience=None,
        required_education_level=None,
        parsing_confidence=0.8,
        pipeline_version="parser-v1",
    )


def test_skeleton_hits_first_placeholder_loudly() -> None:
    # A resume WITH skills proceeds past the short-circuit into STEP 1's placeholder.
    with pytest.raises(NotImplementedError) as exc:
        run_orchestration(_resume(["Python"]), _jd())
    # Fails at the correct step, labeled with its owning phase — not fake data.
    assert "Phase 4.2" in str(exc.value)
    assert "hybrid_scoring" in str(exc.value)


def test_illustrative_short_circuit_returns_low_confidence() -> None:
    # Zero skills → deterministic short-circuit, no exception.
    result = run_orchestration(_resume([]), _jd())
    assert isinstance(result, ScoreResult)
    assert result.final_score == 0
    assert result.confidence_level is ConfidenceLevel.LOW
    assert result.scoring_confidence == 0.0


def test_orchestration_result_conforms_to_score_schema() -> None:
    assert OrchestrationResult is ScoreResult  # reuses the locked schema
    result = run_orchestration(_resume([]), _jd())
    fv = result.feature_vector
    # All five locked FeatureVector petals present and zeroed at short-circuit.
    assert (
        fv.tfidf_score == fv.embedding_score == fv.skill_overlap_pct == 0.0
        and fv.exp_match == fv.edu_match == 0.0
    )
    # pipeline_version is carried through (defaults to the active registry version).
    assert isinstance(result.pipeline_version, str) and result.pipeline_version

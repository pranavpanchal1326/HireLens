"""Tests for the Phase 5.5 bias name-swap harness.

Synthetic fixtures verify the harness MECHANICS (swap correctness, delta stats,
attribution). The stub pipeline deliberately simulates name-sensitivity so we can
confirm the harness DETECTS and reports it — the harness never fixes it.
"""

from __future__ import annotations

from app.schemas.parsing import (
    ExperienceEntry,
    ParsedJobDescription,
    ParsedResume,
)
from app.schemas.scoring import ConfidenceLevel, FeatureVector, ScoreResult
from app.services.evaluation.bias_name_swap_harness import (
    NamePair,
    generate_name_swapped_variant,
    run_name_swap_test,
)


def _resume(description: str, skills=None) -> ParsedResume:
    return ParsedResume(
        raw_text=description,
        skills=skills or ["Python", "SQL"],
        experience=[ExperienceEntry(description=description)],
        education=[],
        total_years_experience=None,
        contact_info_present=False,
        parsing_confidence=1.0,
        parsing_warnings=[],
        pipeline_version="parser-v1",
    )


def _jd() -> ParsedJobDescription:
    return ParsedJobDescription(
        raw_text="jd",
        required_skills=["Python"],
        preferred_skills=[],
        required_years_experience=None,
        required_education_level=None,
        parsing_confidence=1.0,
        pipeline_version="parser-v1",
    )


def _score(final: int, tfidf=0.0, embedding=0.0, skill=0.0) -> ScoreResult:
    return ScoreResult(
        resume_id="r",
        jd_id="j",
        final_score=final,
        feature_vector=FeatureVector(
            tfidf_score=tfidf,
            embedding_score=embedding,
            skill_overlap_pct=skill,
            exp_match=0.0,
            edu_match=0.0,
        ),
        scoring_confidence=0.5,
        confidence_level=ConfidenceLevel.MEDIUM,
        parsing_confidence=1.0,
        pipeline_version="v3-hybrid",
    )


class _EmbeddingBiasPipeline:
    """Simulates embedding-layer name bias: presence of 'Priya' lowers the score
    purely via the embedding feature (a reportable bias to be DETECTED)."""

    def score(self, resume: ParsedResume, jd: ParsedJobDescription) -> ScoreResult:
        text = " ".join([resume.raw_text] + [e.description for e in resume.experience])
        if "Priya" in text:
            return _score(final=40, tfidf=0.5, embedding=0.3, skill=0.5)
        return _score(final=50, tfidf=0.5, embedding=0.5, skill=0.5)


class _NeutralPipeline:
    """Name-insensitive: always returns the same score (no bias)."""

    def score(self, resume: ParsedResume, jd: ParsedJobDescription) -> ScoreResult:
        return _score(final=55, tfidf=0.5, embedding=0.5, skill=0.5)


# --- name swap correctness ---------------------------------------------------


def test_swap_touches_only_name_not_other_content() -> None:
    resume = _resume("John led the data team using Python.", skills=["Python"])
    variant = generate_name_swapped_variant(resume, "John", "Priya")
    assert "Priya led the data team" in variant.experience[0].description
    assert "John" not in variant.experience[0].description
    assert variant.skills == ["Python"]  # unrelated content untouched


def test_swap_is_whole_word_only() -> None:
    resume = _resume("Johnson Corp hired John as an engineer.")
    variant = generate_name_swapped_variant(resume, "John", "Priya")
    # "Johnson" must NOT be altered; standalone "John" must be.
    assert "Johnson Corp" in variant.experience[0].description
    assert "Priya as an engineer" in variant.experience[0].description


def test_swap_no_change_when_name_absent() -> None:
    resume = _resume("A generic engineer with Python skills.")
    variant = generate_name_swapped_variant(resume, "John", "Priya")
    assert variant == resume  # nothing to swap → identical (name_present=False)


# --- paired comparison + delta -----------------------------------------------


def test_detects_and_reports_bias_delta() -> None:
    resumes = [("res1", _resume("John built pipelines in Python."))]
    pairs = (NamePair("John", "Priya", "Western-male ↔ South-Asian-female"),)
    report = run_name_swap_test(resumes, pairs, _jd(), _EmbeddingBiasPipeline())
    assert report.n_trials == 1
    assert report.n_name_present == 1
    # Swapping John→Priya dropped the score 50→40 (bias detected, not hidden).
    assert report.deltas == [-10]
    assert report.mean_abs_delta == 10.0


def test_name_absent_trial_recorded_as_zero_delta() -> None:
    resumes = [("res1", _resume("Generic engineer, Python and SQL."))]
    pairs = (NamePair("John", "Priya", "label"),)
    report = run_name_swap_test(resumes, pairs, _jd(), _EmbeddingBiasPipeline())
    assert report.n_name_present == 0  # name never appeared
    assert report.deltas == [0]  # honest zero, and flagged as not-present


def test_full_distribution_and_per_pair_breakdown_present() -> None:
    resumes = [
        ("res1", _resume("John shipped Python services.")),
        ("res2", _resume("John analyzed SQL data.")),
    ]
    pairs = (
        NamePair("John", "Priya", "pairA"),
        NamePair("John", "Joan", "pairB"),
    )
    report = run_name_swap_test(resumes, pairs, _jd(), _EmbeddingBiasPipeline())
    # Full distribution present (not just a pooled mean).
    assert len(report.deltas) == 4
    assert len(report.per_pair) == 2
    pair_a = next(p for p in report.per_pair if p.pair_label == "pairA")
    assert pair_a.deltas == [-10, -10]  # Priya bias consistent per pair
    pair_b = next(p for p in report.per_pair if p.pair_label == "pairB")
    assert pair_b.deltas == [0, 0]  # 'Joan' not flagged by the stub → no delta


def test_root_cause_attribution_points_to_embedding() -> None:
    resumes = [("res1", _resume("John built pipelines in Python."))]
    pairs = (NamePair("John", "Priya", "label"),)
    report = run_name_swap_test(resumes, pairs, _jd(), _EmbeddingBiasPipeline())
    # Stub moved ONLY the embedding feature (0.5→0.3); attribution must reflect that.
    assert report.attribution.dominant_step == "embedding"
    assert report.trials[0].dominant_step == "embedding"


def test_neutral_pipeline_reports_zero_but_with_distribution() -> None:
    resumes = [("res1", _resume("John built Python services."))]
    pairs = (NamePair("John", "Priya", "label"),)
    report = run_name_swap_test(resumes, pairs, _jd(), _NeutralPipeline())
    assert report.mean_abs_delta == 0.0
    assert report.deltas == [0]  # distribution still shown, not just "≈0"
    assert "DESCRIPTIVE only" in report.significance_note  # small-sample honesty


def test_present_only_stats_undiluted_by_name_absent_zeros() -> None:
    # res1 contains John (bias delta -10); res2 has no John (structural zero).
    resumes = [
        ("res1", _resume("John shipped Python services.")),
        ("res2", _resume("Generic engineer with SQL.")),
    ]
    pairs = (NamePair("John", "Priya", "pairA"),)
    report = run_name_swap_test(resumes, pairs, _jd(), _EmbeddingBiasPipeline())
    # Pooled over ALL trials dilutes the bias: (10 + 0)/2 = 5.0 ...
    assert report.mean_abs_delta == 5.0
    # ...but the bias-relevant present-only view preserves the true signal: 10.0.
    assert report.mean_abs_delta_present == 10.0
    pair_a = report.per_pair[0]
    assert pair_a.mean_abs_delta == 5.0 and pair_a.mean_abs_delta_present == 10.0


def test_small_sample_significance_note_and_caveat() -> None:
    resumes = [("res1", _resume("John built Python services."))]
    report = run_name_swap_test(
        resumes, (NamePair("John", "Priya", "l"),), _jd(), _EmbeddingBiasPipeline()
    )
    assert "PROOF-OF-CONCEPT SCALE" in report.small_sample_caveat
    assert "too small" in report.significance_note

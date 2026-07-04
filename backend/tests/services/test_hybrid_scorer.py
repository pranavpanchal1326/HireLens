"""Tests for the Phase 2.4 combined hybrid scorer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.schemas.parsing import ParsedJobDescription, ParsedResume
from app.services.scoring.hybrid_scorer import (
    DEFAULT_HYBRID_WEIGHTS,
    HybridScorer,
    HybridWeights,
)


def _resume() -> ParsedResume:
    return ParsedResume(
        raw_text="x",
        skills=["Python"],
        experience=[],
        education=[],
        total_years_experience=None,
        contact_info_present=False,
        parsing_confidence=0.9,
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


def _make_scorer(tfidf: float, embedding: float) -> HybridScorer:
    tfidf_scorer = MagicMock()
    tfidf_scorer.score.return_value = tfidf
    cached = MagicMock()
    cached.score.return_value = embedding
    return HybridScorer(tfidf_scorer, cached)


# --- HybridWeights validation ------------------------------------------------


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        HybridWeights(tfidf_weight=0.7, embedding_weight=0.7)
    # Valid case does not raise.
    HybridWeights(tfidf_weight=0.3, embedding_weight=0.7)


# --- final_score arithmetic --------------------------------------------------


def test_default_weights_final_score() -> None:
    scorer = _make_scorer(tfidf=0.8, embedding=0.4)
    result = scorer.compute_hybrid_score("r1", _resume(), "j1", _jd())
    # 0.5*0.8 + 0.5*0.4 = 0.6 -> 60 (round half up).
    assert result.final_score == 60


def test_custom_weights_change_final_score() -> None:
    scorer = _make_scorer(tfidf=0.8, embedding=0.4)
    custom = HybridWeights(tfidf_weight=0.8, embedding_weight=0.2)
    result = scorer.compute_hybrid_score("r1", _resume(), "j1", _jd(), weights=custom)
    # 0.8*0.8 + 0.2*0.4 = 0.72 -> 72.
    assert result.final_score == 72


def test_missing_features_are_zero_and_version_tagged() -> None:
    scorer = _make_scorer(tfidf=0.8, embedding=0.4)
    result = scorer.compute_hybrid_score("r1", _resume(), "j1", _jd())
    fv = result.feature_vector
    assert fv.tfidf_score == 0.8
    assert fv.embedding_score == 0.4
    assert fv.skill_overlap_pct == 0.0
    assert fv.exp_match == 0.0
    assert fv.edu_match == 0.0
    assert result.pipeline_version == "v3-hybrid"


# --- ablation stage scores ---------------------------------------------------


def test_ablation_stages_distinct_and_labeled() -> None:
    scorer = _make_scorer(tfidf=0.8, embedding=0.4)
    stages = scorer.compute_ablation_stage_scores("r1", _resume(), "j1", _jd())

    assert set(stages.keys()) == {"v1-tfidf", "v2-embeddings", "v3-hybrid"}
    assert stages["v1-tfidf"].final_score == 80  # pure tfidf
    assert stages["v2-embeddings"].final_score == 40  # pure embedding
    assert stages["v3-hybrid"].final_score == 60  # 0.5/0.5
    for version, result in stages.items():
        assert result.pipeline_version == version


def test_ablation_v1_matches_direct_pure_tfidf() -> None:
    scorer = _make_scorer(tfidf=0.8, embedding=0.4)
    stages = scorer.compute_ablation_stage_scores("r1", _resume(), "j1", _jd())
    direct_v1 = scorer.compute_hybrid_score(
        "r1",
        _resume(),
        "j1",
        _jd(),
        weights=HybridWeights(tfidf_weight=1.0, embedding_weight=0.0),
    )
    assert stages["v1-tfidf"].final_score == direct_v1.final_score


def test_ablation_computes_raw_scores_only_once() -> None:
    tfidf_scorer = MagicMock()
    tfidf_scorer.score.return_value = 0.8
    cached = MagicMock()
    cached.score.return_value = 0.4
    scorer = HybridScorer(tfidf_scorer, cached)

    scorer.compute_ablation_stage_scores("r1", _resume(), "j1", _jd())

    # Three ScoreResults produced, but each raw scorer invoked exactly once.
    assert tfidf_scorer.score.call_count == 1
    assert cached.score.call_count == 1


# --- interim confidence heuristic --------------------------------------------


def test_interim_confidence_responds_to_signal_agreement() -> None:
    agree = _make_scorer(tfidf=0.8, embedding=0.8).compute_hybrid_score(
        "r1", _resume(), "j1", _jd()
    )
    disagree = _make_scorer(tfidf=0.9, embedding=0.1).compute_hybrid_score(
        "r1", _resume(), "j1", _jd()
    )
    # Strong agreement → higher confidence than strong disagreement.
    assert agree.scoring_confidence > disagree.scoring_confidence


def test_default_weights_constant_is_equal_split() -> None:
    assert DEFAULT_HYBRID_WEIGHTS.tfidf_weight == 0.5
    assert DEFAULT_HYBRID_WEIGHTS.embedding_weight == 0.5

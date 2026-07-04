"""Tests for the pipeline versioning registry (PRD §7.2 / §8.2)."""

from __future__ import annotations

import pytest

from app.core.pipeline_registry import (
    PIPELINE_REGISTRY,
    InvalidPipelineVersionError,
    get_active_pipeline_version,
    get_pipeline_config,
)
from app.schemas.pipeline import PipelineVersion

# The five locked version strings, exactly as PRD §7.2 specifies.
EXPECTED_VERSIONS = {
    "v1-tfidf",
    "v2-embeddings",
    "v3-hybrid",
    "v4-hybrid-rag",
    "v5-full-ml",
}


def test_all_five_versions_exist_exactly() -> None:
    registry_values = {v.value for v in PIPELINE_REGISTRY}
    enum_values = {v.value for v in PipelineVersion}
    assert registry_values == EXPECTED_VERSIONS
    assert enum_values == EXPECTED_VERSIONS


def test_exactly_one_active_version() -> None:
    active = [cfg for cfg in PIPELINE_REGISTRY.values() if cfg.is_active]
    assert len(active) == 1
    # And the enforced accessor agrees.
    assert get_active_pipeline_version() in PipelineVersion


def test_get_pipeline_config_raises_clear_error_for_invalid_version() -> None:
    with pytest.raises(InvalidPipelineVersionError):
        get_pipeline_config("v99-fake")  # type: ignore[arg-type]


def test_v1_tfidf_is_genuinely_tfidf_only() -> None:
    """Guard: v1-tfidf must not secretly leak non-tfidf weight.

    If this fails, the ablation study named 'TF-IDF only' would silently be
    something else, invalidating the graded evaluation deliverable.
    """
    cfg = get_pipeline_config(PipelineVersion.V1_TFIDF)
    assert cfg.enabled_components == ["tfidf"]
    assert cfg.feature_weights is not None
    assert cfg.feature_weights["tfidf_score"] == 1.0
    for feature, weight in cfg.feature_weights.items():
        if feature != "tfidf_score":
            assert weight == 0.0, f"{feature} must be zero for v1-tfidf, got {weight}"


def test_v2_embeddings_is_genuinely_embeddings_only() -> None:
    cfg = get_pipeline_config(PipelineVersion.V2_EMBEDDINGS)
    assert cfg.enabled_components == ["embeddings"]
    assert cfg.feature_weights is not None
    assert cfg.feature_weights["embedding_score"] == 1.0
    for feature, weight in cfg.feature_weights.items():
        if feature != "embedding_score":
            assert weight == 0.0, f"{feature} must be zero for v2-embeddings"


def test_v3_hybrid_weights_are_real_equal_split_and_isolated() -> None:
    """Phase 2.4: v3-hybrid now carries real 0.5/0.5 tfidf/embedding weights, and
    the not-yet-computed features remain zero (no leakage)."""
    cfg = get_pipeline_config(PipelineVersion.V3_HYBRID)
    assert cfg.feature_weights is not None
    assert cfg.feature_weights["tfidf_score"] == 0.5
    assert cfg.feature_weights["embedding_score"] == 0.5
    for feature in ("skill_overlap_pct", "exp_match", "edu_match"):
        assert cfg.feature_weights[feature] == 0.0
    assert round(sum(cfg.feature_weights.values()), 6) == 1.0

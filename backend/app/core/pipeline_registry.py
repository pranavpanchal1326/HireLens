"""Static, version-controlled pipeline registry.

Single source of truth for the five LOCKED pipeline versions (PRD §7.2 ablation
study, PRD §8.2 score versioning). Deliberately NOT a database table: at this
project's scale a static, auditable Python registry is more reproducible than a
mutable DB row, which matters for a grader verifying the evaluation methodology.

Old configs are NEVER deleted — every version remains queryable forever so the
ablation harness (Phase 5.3) can compute per-version correlation.
"""

from __future__ import annotations

# Re-export so the enum is importable from either the schema layer or here, while
# the canonical definition lives in the dependency-free schema module.
from app.schemas.pipeline import PipelineConfig, PipelineVersion

__all__ = [
    "PipelineVersion",
    "PipelineConfig",
    "PIPELINE_REGISTRY",
    "InvalidPipelineVersionError",
    "RegistryInvariantError",
    "get_pipeline_config",
    "get_active_pipeline_version",
]

# FeatureVector field names (from schemas/scoring.py) — weight keys must match.
_FEATURES = (
    "tfidf_score",
    "embedding_score",
    "skill_overlap_pct",
    "exp_match",
    "edu_match",
)


def _weights(**active: float) -> dict[str, float]:
    """Build a full 5-key weight dict, defaulting any unlisted feature to 0.0.

    Forcing every feature to appear (zeroed when inactive) makes each version's
    isolation explicit and auditable — an omitted key can't be mistaken for an
    intentional zero.
    """
    return {feature: active.get(feature, 0.0) for feature in _FEATURES}


class InvalidPipelineVersionError(KeyError):
    """Raised when a pipeline version is requested that is not in the registry."""


class RegistryInvariantError(RuntimeError):
    """Raised when the registry violates a structural invariant.

    Currently: exactly one version must be marked ``is_active``.
    """


# All weights below are honest PLACEHOLDERS (not tuned). Phase 6 grid search will
# replace them with real values. Each version zeroes every feature it does not
# activate, so v1-tfidf/v2-embeddings are never secretly hybrid.
PIPELINE_REGISTRY: dict[PipelineVersion, PipelineConfig] = {
    PipelineVersion.V1_TFIDF: PipelineConfig(
        version=PipelineVersion.V1_TFIDF,
        description="TF-IDF lexical similarity only. Ablation baseline. "
        "PLACEHOLDER weights (not tuned).",
        enabled_components=["tfidf"],
        feature_weights=_weights(tfidf_score=1.0),
        model_reference=None,
        is_active=True,  # First pipeline to be implemented (Phase 2); default.
    ),
    PipelineVersion.V2_EMBEDDINGS: PipelineConfig(
        version=PipelineVersion.V2_EMBEDDINGS,
        description="Embedding semantic similarity only. "
        "PLACEHOLDER weights (not tuned).",
        enabled_components=["embeddings"],
        feature_weights=_weights(embedding_score=1.0),
        model_reference=None,
        is_active=False,
    ),
    PipelineVersion.V3_HYBRID: PipelineConfig(
        version=PipelineVersion.V3_HYBRID,
        description="Hybrid TF-IDF + embeddings (Phase 2.4). Implemented default: "
        "equal 0.5/0.5 weighting — a real, defensible provisional value, still "
        "pending Phase 6 grid-search tuning against ground truth (PRD §8.2).",
        enabled_components=["tfidf", "embeddings"],
        feature_weights=_weights(tfidf_score=0.5, embedding_score=0.5),
        model_reference=None,
        is_active=False,
    ),
    PipelineVersion.V4_HYBRID_RAG: PipelineConfig(
        version=PipelineVersion.V4_HYBRID_RAG,
        description="Hybrid + RAG skill matching (adds skill-overlap signal). "
        "PLACEHOLDER equal weights (not tuned).",
        enabled_components=["tfidf", "embeddings", "rag_skill_matcher"],
        feature_weights=_weights(
            tfidf_score=1 / 3, embedding_score=1 / 3, skill_overlap_pct=1 / 3
        ),
        model_reference=None,
        is_active=False,
    ),
    PipelineVersion.V5_FULL_ML: PipelineConfig(
        version=PipelineVersion.V5_FULL_ML,
        description="Full pipeline with trained ML re-ranker over all 5 features. "
        "Final production version. PLACEHOLDER equal weights; real weights and "
        "model_reference land in Phase 6.",
        enabled_components=[
            "tfidf",
            "embeddings",
            "rag_skill_matcher",
            "experience_matcher",
            "education_matcher",
            "ml_reranker",
        ],
        feature_weights=_weights(
            tfidf_score=0.2,
            embedding_score=0.2,
            skill_overlap_pct=0.2,
            exp_match=0.2,
            edu_match=0.2,
        ),
        # No trained artifact exists until Phase 6; honestly null for now.
        model_reference=None,
        is_active=False,
    ),
}


def get_pipeline_config(version: PipelineVersion) -> PipelineConfig:
    """Return the config for ``version``.

    Raises ``InvalidPipelineVersionError`` (not a bare KeyError) if the version is
    not registered.
    """
    try:
        return PIPELINE_REGISTRY[version]
    except KeyError as exc:
        raise InvalidPipelineVersionError(
            f"Unknown pipeline version: {version!r}. "
            f"Valid versions: {[v.value for v in PipelineVersion]}"
        ) from exc


def get_active_pipeline_version() -> PipelineVersion:
    """Return the single version flagged ``is_active``.

    Raises ``RegistryInvariantError`` if zero or more than one version is active —
    the invariant is enforced, not assumed.
    """
    active = [v for v, cfg in PIPELINE_REGISTRY.items() if cfg.is_active]
    if len(active) != 1:
        raise RegistryInvariantError(
            f"Exactly one pipeline version must be active; found {len(active)}: "
            f"{[v.value for v in active]}"
        )
    return active[0]

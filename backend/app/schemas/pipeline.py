"""Pipeline versioning data contracts.

Defines the locked set of pipeline version identifiers and the config shape that
describes each one.

Maps to: PRD §7.2 (mandatory ablation study across exactly these five stages) and
PRD §8.2 (score versioning — every ScoreResult.pipeline_version must be one of
these values).

The ``PipelineVersion`` enum lives here (not in ``core/pipeline_registry.py``) so
that schema modules stay dependency-free and independently importable, per the
Phase 0.2 no-circular-imports rule. ``pipeline_registry`` re-exports it, so it is
still the single source of truth and importable everywhere.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class PipelineVersion(str, Enum):
    """The five LOCKED pipeline version identifiers (PRD §7.2 ablation stages).

    Do not rename, reorder, or add a sixth member. This enum is the single source
    of truth for valid ``pipeline_version`` strings across the whole system,
    preventing typo-fragmentation (e.g. "v1-tfidf" vs "v1_tfidf") from silently
    splitting the evaluation data.
    """

    V1_TFIDF = "v1-tfidf"
    V2_EMBEDDINGS = "v2-embeddings"
    V3_HYBRID = "v3-hybrid"
    V4_HYBRID_RAG = "v4-hybrid-rag"
    V5_FULL_ML = "v5-full-ml"


class PipelineConfig(BaseModel):
    """Reproducible configuration for a single pipeline version.

    Maps to: PRD §7.2 / §8.2. ``feature_weights`` keys must match ``FeatureVector``
    field names from ``schemas/scoring.py``. Weights are honest PLACEHOLDERS until
    Phase 6 grid search populates tuned values — each version's weights reflect
    only the components it actually activates (e.g. v1-tfidf gives zero weight to
    every non-tfidf feature), so the ablation study stays scientifically valid.
    """

    version: PipelineVersion
    description: str
    enabled_components: list[str]
    feature_weights: dict[str, float] | None = None
    model_reference: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    is_active: bool = False

"""Ground-truth dataset schema (NEW Phase 5.1 artifact).

IMPORTANT: these models are a NEW artifact defined here in Phase 5.1 for the
evaluation dataset. They are NOT a modification of the Phase 0.2 canonical schemas
(ParsedResume/ScoreResult, etc.), which remain untouched. Phase 5.2 (metrics),
5.3 (ablation), 5.4 (CV), and 6.2 (training) consume THIS schema.

Honesty (Design Blueprint P3): rater scores are preserved individually and
divergence is flagged, never smoothed into a falsely-confident single number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

CaseType = Literal["clear_fit", "clear_gap", "ambiguous"]
PairStatus = Literal["awaiting_raters", "reconciled"]

GROUND_TRUTH_SCHEMA_VERSION = "gt-v1"


class RaterScore(BaseModel):
    """One human rater's blind score + justification for a pair."""

    rater_id: str
    score: float = Field(ge=0.0, le=100.0)
    justification: str


class GroundTruthPair(BaseModel):
    """One resume/JD pair with all rater input and its reconciliation."""

    pair_id: str
    resume_id: str
    jd_id: str
    case_type: CaseType
    rater_scores: list[RaterScore] = Field(default_factory=list)
    reconciled_score: float | None = None  # None until rated + reconciled
    inter_rater_range: float | None = None  # per-pair max-min (divergence size)
    divergence_flag: bool = False  # True when raters disagree beyond threshold
    status: PairStatus = "awaiting_raters"


class GroundTruthDataset(BaseModel):
    """The full evaluation ground-truth set (flat-file persisted)."""

    version: str = GROUND_TRUTH_SCHEMA_VERSION
    pairs: list[GroundTruthPair] = Field(default_factory=list)
    n_raters: int | None = None
    # Overall inter-rater reliability (mean pairwise Pearson across raters).
    overall_inter_rater_agreement: float | None = None
    notes: str = ""


def save_dataset(dataset: GroundTruthDataset, output_path: str) -> None:
    """Persist as a single pretty JSON file (versioned, human-inspectable)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dataset.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_dataset(input_path: str) -> GroundTruthDataset:
    """Load a persisted ground-truth dataset for Phase 5.2+ to consume directly."""
    raw = Path(input_path).read_text(encoding="utf-8")
    return GroundTruthDataset.model_validate_json(raw)

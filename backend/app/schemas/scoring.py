"""Scoring data contracts.

Canonical shapes for the feature vector, skill matches, gaps, and the final
score result.

Maps to: PRD §4 (feature vector definition), PRD §8.2 (score versioning + parsing
vs scoring confidence traceability), Design Blueprint §6.2 (the five-petal
aperture-bloom rendering contract), §10.6 (exact vs semantic match "≈"), and §12
(gaps framed as to-dos, never verdicts).

Produced by: Phase 2/6 (scorer + trained model). Consumed by: Phase 8 (frontend).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class FeatureVector(BaseModel):
    """The five normalized scoring features.

    LOCKED CONTRACT — Maps to: PRD §4 (feature vector) and Design Blueprint §6.2.
    Field ORDER and NAMING are rendered literally as the five petals of the
    aperture-bloom UI signature: [tfidf_score, embedding_score, skill_overlap_pct,
    exp_match, edu_match]. Do NOT reorder, rename, add, or remove fields without
    flagging it as a breaking design-system change. All values are 0.0-1.0.
    """

    tfidf_score: float = Field(ge=0.0, le=1.0)
    embedding_score: float = Field(ge=0.0, le=1.0)
    skill_overlap_pct: float = Field(ge=0.0, le=1.0)
    exp_match: float = Field(ge=0.0, le=1.0)
    edu_match: float = Field(ge=0.0, le=1.0)


class ConfidenceLevel(str, Enum):
    """Bucketed confidence label for display.

    Maps to: Design Blueprint §6.3 (confidence surfacing).
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SkillMatch(BaseModel):
    """A single matched skill pair between resume and JD.

    Maps to: Design Blueprint §10.6 — ``match_type`` distinguishes exact matches
    from RAG-derived semantic matches (e.g. "led team" ≈ "people management"),
    which the frontend renders with the "≈" symbol.
    """

    resume_skill: str
    jd_skill: str
    match_type: Literal["exact", "semantic"]
    similarity_score: float = Field(ge=0.0, le=1.0)


class GapItem(BaseModel):
    """A missing skill framed as an actionable next step.

    Maps to: Design Blueprint §12 voice rule — "Gaps are to-dos, not verdicts."
    ``suggested_action`` must contain only action-framed language, never
    deficiency-framed language.
    """

    missing_skill: str
    suggested_action: str


class ScoreResult(BaseModel):
    """Canonical scoring-result contract.

    Maps to: PRD §4 (feature vector), PRD §8.2 (score versioning + parsing/scoring
    confidence traceability). ``scoring_confidence`` is DISTINCT from
    ``parsing_confidence``; both are carried so a low score is always traceable to
    "couldn't read it" vs "genuine mismatch".
    """

    score_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    resume_id: str
    jd_id: str
    final_score: int = Field(ge=0, le=100)
    feature_vector: FeatureVector
    scoring_confidence: float = Field(ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel
    parsing_confidence: float = Field(ge=0.0, le=1.0)
    matched_skills: list[SkillMatch] = Field(default_factory=list)
    gaps: list[GapItem] = Field(default_factory=list)
    feature_importance: dict[str, float] | None = None
    pipeline_version: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

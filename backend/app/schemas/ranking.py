"""Ranking data contracts.

Canonical shapes for recruiter-side candidate ranking.

Maps to: PRD §3.2 / §7.4 (configurable weights, blind-mode) and Design Blueprint
§10.7 / §11.3 (weight overrides, anonymized display).

Produced by: Phase 7 (ranking/orchestration). Consumed by: Phase 8 (recruiter view).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.scoring import ScoreResult


class RankedCandidate(BaseModel):
    """One candidate's position within a ranked list.

    Maps to: PRD §7.4 / Design Blueprint §11.3 — ``anonymized_display_name``
    supports recruiter blind-mode.
    """

    rank: int = Field(ge=1)
    candidate_id: str
    score_result: ScoreResult
    anonymized_display_name: str | None = None


class RankingRequest(BaseModel):
    """Request to rank a set of resumes against one JD.

    Maps to: PRD §3.2 / Design Blueprint §10.7 — ``weight_overrides`` keys must
    match ``FeatureVector`` field names.
    """

    jd_id: str
    resume_ids: list[str]
    weight_overrides: dict[str, float] | None = None


class RankingResult(BaseModel):
    """Canonical ranking-result contract.

    Maps to: PRD §4 (ranking output) and PRD §8.2 (pipeline versioning).
    """

    jd_id: str
    ranked_candidates: list[RankedCandidate] = Field(default_factory=list)
    pipeline_version: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

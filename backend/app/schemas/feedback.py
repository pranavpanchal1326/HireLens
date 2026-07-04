"""Feedback and ground-truth data contracts.

Canonical shapes for human labels (model training/evaluation) and post-hoc
outcome feedback.

Maps to: PRD §6 (self-built ground truth with multi-rater reconciliation) and
the feedback loop feeding Phase 6 model training.

Produced by: Phase 5 (ground-truth build) / recruiter feedback. Consumed by:
Phase 6 (model training + evaluation).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class GroundTruthLabel(BaseModel):
    """A single human-provided relevance label for a resume/JD pair.

    Maps to: PRD §6 — ``rater_id`` supports the multi-rater reconciliation
    approach used to build the ground-truth set.
    """

    pair_id: str
    resume_id: str
    jd_id: str
    human_rank_or_score: float
    rater_id: str
    notes: str | None = None


class FeedbackSubmission(BaseModel):
    """Post-hoc real-world outcome for a produced score.

    Maps to: PRD §6 feedback loop — closes the loop between predicted score and
    actual hiring outcome for continued model evaluation.
    """

    score_id: str
    actual_outcome: Literal["hired", "rejected", "interviewed", "no_action"]
    recruiter_id: str
    submitted_at: datetime

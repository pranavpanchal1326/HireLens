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
from enum import Enum
from typing import Literal, Union

from pydantic import BaseModel, Field, field_validator


class RecruiterOutcome(str, Enum):
    """Real-world outcome categories for candidate evaluation."""
    HIRED = "hired"
    REJECTED = "rejected"
    INTERVIEWED = "interviewed"
    NO_ACTION = "no_action"


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


class RaterFeedbackRequest(BaseModel):
    """Request model for rater fit-score submission."""

    feedback_type: Literal["rater"]
    pair_id: str
    resume_id: str
    jd_id: str
    score: float = Field(..., ge=0.0, le=100.0, description="Fit score from 0.0 to 100.0")
    rater_id: str
    justification: str
    case_type: Literal["clear_fit", "clear_gap", "ambiguous"] = "ambiguous"

    @field_validator("justification")
    @classmethod
    def validate_justification(cls, v: str) -> str:
        """Enforce non-empty, non-numeric, reasoning justification."""
        v = v.strip()
        if not v:
            raise ValueError("Justification cannot be empty or whitespace only.")
        if v.isdigit() or (v.replace(".", "", 1).isdigit() and v.count(".") <= 1):
            raise ValueError("Justification cannot be a numeric value.")
        if len(v) < 5:
            raise ValueError("Justification must be at least 5 characters long to ensure reasoning is provided.")
        return v


class RecruiterFeedbackRequest(BaseModel):
    """Request model for recruiter hiring-outcome feedback."""

    feedback_type: Literal["recruiter"]
    score_id: str
    actual_outcome: RecruiterOutcome
    recruiter_id: str


# Discriminated union for routing requests based on feedback_type
FeedbackRequest = Union[RaterFeedbackRequest, RecruiterFeedbackRequest]


class RaterFeedbackResponse(BaseModel):
    """Response returned upon successfully storing rater feedback."""

    stored_id: str
    feedback_type: Literal["rater"]
    status: str
    details: dict
    progress: dict
    schema_gap_note: str | None = "SCHEMA GAP: Response shape is not part of Phase 0.2 canonical schemas."


class RecruiterFeedbackResponse(BaseModel):
    """Response returned upon successfully storing recruiter outcome feedback."""

    stored_id: str
    feedback_type: Literal["recruiter"]
    status: str
    details: dict
    schema_gap_note: str | None = "SCHEMA GAP: Response shape is not part of Phase 0.2 canonical schemas."


FeedbackResponse = Union[RaterFeedbackResponse, RecruiterFeedbackResponse]


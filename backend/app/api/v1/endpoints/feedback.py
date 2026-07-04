# ruff: noqa: E501
"""POST /feedback endpoint implementation (Phase 7.5).

Exposes a live HTTP route for logging ground-truth signals and recruiter outcomes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, status, Depends
from app.core.auth import RecruiterAccount, get_current_recruiter

from app.schemas.feedback import (
    FeedbackRequest,
    FeedbackResponse,
    RaterFeedbackRequest,
    RaterFeedbackResponse,
    RecruiterFeedbackRequest,
    RecruiterFeedbackResponse,
)
from app.services.evaluation.ground_truth_schema import (
    GroundTruthDataset,
    GroundTruthPair,
    RaterScore,
    load_dataset,
    save_dataset,
)
from app.services.evaluation.reconciliation import reconcile_dataset

logger = logging.getLogger(__name__)

router = APIRouter()

# Global write lock to serialize file modifications and prevent concurrent write corruption
_write_lock = asyncio.Lock()

# In-memory store for idempotency keys to prevent duplicate processing of retries
_idempotency_keys: dict[str, dict[str, Any]] = {}

# Resolved paths (can be overridden in testing)
REPO_ROOT = Path(__file__).resolve().parents[5]
GT_DIR = REPO_ROOT / "data" / "ground_truth"
DATASET_PATH = GT_DIR / "ground_truth_dataset.json"
RECRUITER_PATH = GT_DIR / "recruiter_feedback.json"
AUDIT_LOG_PATH = GT_DIR / "feedback_audit_log.jsonl"


def check_ground_truth_collection_progress(dataset: GroundTruthDataset) -> dict[str, int]:
    """Calculate the current multi-rater coverage progress across all pairs in the dataset.

    Full coverage requires at least 3 distinct rater scores per pair.
    """
    pairs = dataset.pairs
    covered_count = sum(1 for p in pairs if len(p.rater_scores) >= 3)
    needing_count = sum(1 for p in pairs if len(p.rater_scores) < 3)

    # Target is set to the total curated pairs in the dataset, fallback to 25 if empty
    total_target = len(pairs) if len(pairs) > 0 else 25

    return {
        "pairs_with_full_rater_coverage": covered_count,
        "pairs_still_needing_raters": needing_count,
        "total_target": total_target,
    }


@router.post("", response_model=FeedbackResponse, status_code=status.HTTP_200_OK)
async def post_feedback(
    request: Request,
    payload: FeedbackRequest,
    x_idempotency_key: str | None = Header(None, alias="X-Idempotency-Key"),
    current_recruiter: RecruiterAccount = Depends(get_current_recruiter),
) -> Any:
    """POST /feedback endpoint to log ground-truth relevance signals or real-world outcomes.

    Supports:
      1. Rater fit-score submissions: Appends to ground_truth_dataset.json and triggers live reconciliation.
      2. Recruiter outcome submissions: Appends to recruiter_feedback.json for hit-rate tracking.

    Ensures idempotency via HTTP header X-Idempotency-Key or natural keys.
    Logs every request with full provenance inside feedback_audit_log.jsonl.
    """
    # Ensure directories exist
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECRUITER_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Check header-based idempotency cache
    if x_idempotency_key and x_idempotency_key in _idempotency_keys:
        logger.info(f"Idempotency hit for key: {x_idempotency_key}")
        cached = _idempotency_keys[x_idempotency_key]
        if cached["feedback_type"] == "rater":
            return RaterFeedbackResponse(**cached["response"])
        else:
            return RecruiterFeedbackResponse(**cached["response"])

    # Read raw request payload for provenance logging
    try:
        raw_payload = await request.json()
    except Exception:
        raw_payload = payload.model_dump()

    # Process the request under a global lock to prevent write race-conditions
    async with _write_lock:
        if isinstance(payload, RaterFeedbackRequest):
            # Load dataset
            try:
                if DATASET_PATH.exists():
                    dataset = load_dataset(str(DATASET_PATH))
                else:
                    dataset = GroundTruthDataset(pairs=[])
            except Exception as e:
                logger.warning(f"Error loading ground truth dataset: {e}. Creating a new one.")
                dataset = GroundTruthDataset(pairs=[])

            # Search for existing pair
            pair = next((p for p in dataset.pairs if p.pair_id == payload.pair_id), None)
            status_action = "created"

            if pair:
                # Check for existing rating by the same rater (natural key uniqueness check)
                existing_score = next((rs for rs in pair.rater_scores if rs.rater_id == payload.rater_id), None)
                if existing_score:
                    if existing_score.score == payload.score and existing_score.justification == payload.justification:
                        # Exact duplicate - natural key idempotency match
                        status_action = "duplicate"
                    else:
                        # Correction/Update
                        existing_score.score = payload.score
                        existing_score.justification = payload.justification
                        status_action = "updated"
                else:
                    # New rater rating for an existing pair
                    pair.rater_scores.append(
                        RaterScore(rater_id=payload.rater_id, score=payload.score, justification=payload.justification)
                    )
                    status_action = "created"
            else:
                # Completely new pair
                new_pair = GroundTruthPair(
                    pair_id=payload.pair_id,
                    resume_id=payload.resume_id,
                    jd_id=payload.jd_id,
                    case_type=payload.case_type,
                    rater_scores=[
                        RaterScore(rater_id=payload.rater_id, score=payload.score, justification=payload.justification)
                    ],
                )
                dataset.pairs.append(new_pair)
                status_action = "created"

            # Reconcile and save if we performed a write
            if status_action != "duplicate":
                dataset = reconcile_dataset(dataset)
                save_dataset(dataset, str(DATASET_PATH))

            progress = check_ground_truth_collection_progress(dataset)

            response_data = RaterFeedbackResponse(
                stored_id=payload.pair_id,
                feedback_type="rater",
                status=status_action,
                details={
                    "rater_id": payload.rater_id,
                    "score": payload.score,
                    "justification": payload.justification,
                    "case_type": payload.case_type,
                },
                progress=progress,
            )

        elif isinstance(payload, RecruiterFeedbackRequest):
            # Load recruiter outcomes
            outcomes: list[dict[str, Any]] = []
            if RECRUITER_PATH.exists():
                try:
                    with RECRUITER_PATH.open("r", encoding="utf-8") as f:
                        outcomes = json.load(f)
                except Exception as e:
                    logger.warning(f"Error loading recruiter outcomes: {e}. Initializing empty list.")
                    outcomes = []

            if payload.recruiter_id != current_recruiter.recruiter_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: Cannot log outcomes on behalf of another recruiter."
                )

            # Check for existing score_id outcome
            existing_outcome = next((o for o in outcomes if o["score_id"] == payload.score_id), None)
            status_action = "created"

            if existing_outcome:
                if existing_outcome["recruiter_id"] != current_recruiter.recruiter_id:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Forbidden: Recruiter outcome ownership validation failed."
                    )
                if existing_outcome["actual_outcome"] == payload.actual_outcome.value and existing_outcome["recruiter_id"] == payload.recruiter_id:
                    # Natural key duplicate
                    status_action = "duplicate"
                else:
                    # Correction
                    existing_outcome["actual_outcome"] = payload.actual_outcome.value
                    existing_outcome["recruiter_id"] = payload.recruiter_id
                    existing_outcome["submitted_at"] = datetime.now(UTC).isoformat()
                    status_action = "updated"
            else:
                outcomes.append({
                    "score_id": payload.score_id,
                    "actual_outcome": payload.actual_outcome.value,
                    "recruiter_id": payload.recruiter_id,
                    "submitted_at": datetime.now(UTC).isoformat(),
                })
                status_action = "created"

            # Save if modified
            if status_action != "duplicate":
                with RECRUITER_PATH.open("w", encoding="utf-8") as f:
                    json.dump(outcomes, f, indent=2)

            response_data = RecruiterFeedbackResponse(
                stored_id=payload.score_id,
                feedback_type="recruiter",
                status=status_action,
                details={
                    "recruiter_id": payload.recruiter_id,
                    "actual_outcome": payload.actual_outcome.value,
                },
            )

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid feedback type.",
            )

        # Append record to audit log for provenance tracking
        audit_record = {
            "id": str(uuid.uuid4()),
            "feedback_type": payload.feedback_type,
            "submitter_identity": payload.rater_id if isinstance(payload, RaterFeedbackRequest) else payload.recruiter_id,
            "role": "recruiter" if isinstance(payload, RecruiterFeedbackRequest) else "rater",
            "timestamp": datetime.now(UTC).isoformat(),
            "raw_payload": raw_payload,
            "normalized_payload": payload.model_dump(),
        }

        with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(audit_record) + "\n")

        # Cache response if idempotency key is provided
        if x_idempotency_key:
            _idempotency_keys[x_idempotency_key] = {
                "feedback_type": payload.feedback_type,
                "response": response_data.model_dump(),
            }

        return response_data

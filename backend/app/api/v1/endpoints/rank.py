# ruff: noqa: B008, E501
"""POST /rank endpoint — batch candidate ranking (one JD against N resumes).

R5 delivery model:
  - Batches <= SYNC_THRESHOLD are scored SYNCHRONOUSLY and return the original
    RankResponse immediately (unchanged, verified contract — existing behavior).
  - Larger batches (previously rejected) are enqueued as a background job; POST
    returns {job_id, status, estimated_items} and the recruiter polls
    GET /rank/{job_id}/status for progress and the SAME final RankResponse shape.
Both paths call the one shared run_batch_ranking (→ the confirmed R6 scoring path),
so sync and async can never drift.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.api.v1.endpoints.score import (
    get_orchestrator_tools,
    get_pipeline_maturity_status,
)
from app.api.v1.guardrails import validate_batch_size
from app.core.auth import RecruiterAccount, get_current_recruiter
from app.schemas.parsing import ParsedJobDescription, ParsedResume
from app.schemas.ranking import RankingResult
from app.services.orchestration.agent_orchestrator import OrchestratorTools
from app.services.ranking.batch_ranking import _resolve_jd, run_batch_ranking
from app.services.ranking.job_store import JobStatus, RankJobStore

logger = logging.getLogger(__name__)

router = APIRouter()

# Batches at or below this run synchronously (~0.5s/resume → ~25s worst case,
# under the 30s gateway timeout). Larger batches go async to avoid the timeout.
SYNC_THRESHOLD = 50
# Hard ceiling on a single async batch (latency/memory sanity bound).
MAX_BATCH_SIZE = 1000
# Backpressure: max concurrently non-terminal jobs before we shed load honestly.
MAX_ACTIVE_JOBS = 4
# A job stuck 'processing' longer than this is reaped as failed (dead-letter).
JOB_TIMEOUT_SECONDS = 600


# ============================ SCHEMAS =======================================


class ResumeInput(BaseModel):
    candidate_id: str = Field(..., description="Unique recruiter-defined identifier for this candidate row.")
    parsed_resume: ParsedResume | None = None
    raw_resume_text: str | None = None


class RankRequest(BaseModel):
    parsed_jd: ParsedJobDescription | None = None
    raw_jd_text: str | None = None
    resumes: list[ResumeInput] = Field(
        ..., min_length=1, max_length=MAX_BATCH_SIZE,
        description=f"Resumes to score and rank. Max batch size is {MAX_BATCH_SIZE}.",
    )


class RankResponse(BaseModel):
    """Synchronous ranking result — also the exact shape returned by the async
    status endpoint on completion (client-facing contract, unchanged)."""

    ranking_result: RankingResult
    pipeline_maturity: dict[str, Any]
    total_submitted: int
    total_successful: int
    total_failed: int
    failures: list[dict[str, str]] = Field(default_factory=list)


class RankJobEnvelope(BaseModel):
    """Immediate response for an async (large) batch submission."""

    job_id: str
    status: str
    estimated_items: int


class RankJobStatus(BaseModel):
    """Polling response. ``result`` is populated (RankResponse shape) once complete."""

    job_id: str
    status: str
    completed: int
    total: int
    result: RankResponse | None = None


# ============================ JOB STORE DEPENDENCY ==========================

_JOB_STORE: RankJobStore | None = None


def get_job_store() -> RankJobStore:
    """Singleton file-backed job store (R3-consistent). Overridable in tests."""
    global _JOB_STORE
    if _JOB_STORE is None:
        repo_root = Path(__file__).resolve().parents[5]
        _JOB_STORE = RankJobStore(repo_root / "data" / "processed" / "rank_jobs.json")
    return _JOB_STORE


def _count_active(store: RankJobStore) -> int:
    # Small POC scale: read the file once. (Not O(1), but batches are rare events.)
    data = store._load()  # noqa: SLF001 — deliberate: store owns the file, this is read-only
    return sum(1 for r in data.values() if r["status"] not in JobStatus.TERMINAL)


def _process_job(
    job_id: str, request: RankRequest, tools: OrchestratorTools,
    maturity: dict[str, Any], store: RankJobStore,
) -> None:
    """Background worker: run the batch, streaming progress into the job store."""
    store.update(job_id, status=JobStatus.PROCESSING)
    try:
        result = run_batch_ranking(
            request, tools, maturity,
            progress_cb=lambda done, total: store.update(job_id, completed=done),
        )
        store.update(
            job_id, status=JobStatus.COMPLETE,
            completed=result["total_submitted"], result=result,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ranking job %s failed.", job_id)
        store.update(job_id, status=JobStatus.FAILED, result={"error": str(exc)})


# ============================ ROUTE HANDLERS ================================


@router.post("/rank", response_model=None)
async def rank_candidates(
    request: RankRequest,
    background_tasks: BackgroundTasks,
    response: Response,
    tools: OrchestratorTools = Depends(get_orchestrator_tools),
    current_recruiter: RecruiterAccount = Depends(get_current_recruiter),
    store: RankJobStore = Depends(get_job_store),
) -> RankResponse | RankJobEnvelope:
    """Rank a batch of candidates against one JD.

    <= SYNC_THRESHOLD → synchronous RankResponse. Larger → async job + polling.
    """
    total = len(request.resumes)
    logger.info(
        "Recruiter '%s' (account: '%s') initiated batch rank for %d candidates.",
        current_recruiter.recruiter_id, current_recruiter.account_id, total,
    )
    vr = validate_batch_size(total, MAX_BATCH_SIZE)
    if not vr.is_valid:
        raise HTTPException(status_code=vr.http_status, detail=vr.error_detail)

    # Resolve + validate the JD once, up front, so an invalid JD is a clean 400 on
    # BOTH the sync and async paths (never a 500, never a failed async job).
    try:
        request.parsed_jd = _resolve_jd(request)
        request.raw_jd_text = None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    maturity = get_pipeline_maturity_status()

    # --- Synchronous path (unchanged contract) ---
    if total <= SYNC_THRESHOLD:
        result = await run_in_threadpool(run_batch_ranking, request, tools, maturity)
        return RankResponse.model_validate(result)

    # --- Async path (large batches) ---
    if _count_active(store) >= MAX_ACTIVE_JOBS:
        response.headers["Retry-After"] = "30"
        raise HTTPException(
            status_code=503,
            detail="Ranking workers are busy. Please retry in ~30 seconds.",
            headers={"Retry-After": "30"},
        )

    job_id = str(uuid.uuid4())
    store.create(job_id, owner_account_id=current_recruiter.account_id, total=total)
    background_tasks.add_task(_process_job, job_id, request, tools, maturity, store)
    response.status_code = 202
    return RankJobEnvelope(job_id=job_id, status=JobStatus.QUEUED, estimated_items=total)


@router.get("/rank/{job_id}/status", response_model=RankJobStatus)
async def rank_job_status(
    job_id: str,
    current_recruiter: RecruiterAccount = Depends(get_current_recruiter),
    store: RankJobStore = Depends(get_job_store),
) -> RankJobStatus:
    """Poll a ranking job. Auth-guarded and company-scoped (IDOR-safe)."""
    record = store.get(job_id)
    # 404 for missing OR not-owned — never leak another company's job existence.
    if record is None or record["owner_account_id"] != current_recruiter.account_id:
        raise HTTPException(status_code=404, detail="Ranking job not found.")

    # Dead-letter reaper: a job stuck 'processing' past the timeout is failed.
    if record["status"] == JobStatus.PROCESSING:
        started = datetime.fromisoformat(record["updated_at"])
        if (datetime.now(UTC) - started).total_seconds() > JOB_TIMEOUT_SECONDS:
            store.update(job_id, status=JobStatus.FAILED,
                         result={"error": "Job timed out and was reaped."})
            record = store.get(job_id)

    result_payload = None
    if record["status"] == JobStatus.COMPLETE and isinstance(record.get("result"), dict):
        result_payload = RankResponse.model_validate(record["result"])

    return RankJobStatus(
        job_id=job_id, status=record["status"],
        completed=record["completed"], total=record["total"], result=result_payload,
    )

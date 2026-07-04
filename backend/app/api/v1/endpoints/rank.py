# ruff: noqa: B008, E501
"""POST /rank endpoint implementation (Phase 7.4).

Exposes a live HTTP route for batch candidate ranking (one JD against N resumes).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.api.v1.endpoints.score import (
    get_orchestrator_tools,
    get_pipeline_maturity_status,
)
from app.schemas.parsing import ExtractionResult, ParsedJobDescription, ParsedResume
from app.schemas.ranking import RankedCandidate, RankingResult
from app.schemas.scoring import ScoreResult
from app.services.orchestration.agent_orchestrator import (
    OrchestratorTools,
    run_orchestration,
)
from app.services.structuring.nlp_pipeline import (
    structure_job_description,
    structure_resume,
)
from app.api.v1.guardrails import (
    validate_batch_size,
    validate_text_input,
    detect_content_quality,
)
from app.core.auth import RecruiterAccount, get_current_recruiter

logger = logging.getLogger(__name__)

router = APIRouter()

# Max Synchronous Batch Size Limit: 50 candidates
# Latency-budget justification arithmetic:
#   - Single resume parse & score pass takes ~0.3s - 0.5s of CPU time.
#   - At 0.5s/resume, a batch of 50 takes 25.0s, staying safely below the typical 30.0s web gateway timeout.
#   - Enforced natively at payload parsing time via Pydantic constraints.
MAX_BATCH_SIZE = 50


# ============================ SCHEMAS =======================================


class ResumeInput(BaseModel):
    """Container for a single candidate row, accepting either pre-parsed JSON or raw text."""

    candidate_id: str = Field(
        ...,
        description="Unique recruiter-defined identifier for this candidate row.",
    )
    parsed_resume: ParsedResume | None = None
    raw_resume_text: str | None = None


class RankRequest(BaseModel):
    """Payload accepting one job description and N resumes for batch ranking."""

    parsed_jd: ParsedJobDescription | None = None
    raw_jd_text: str | None = None
    resumes: list[ResumeInput] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"List of resumes to score and rank. Maximum batch size is {MAX_BATCH_SIZE}.",
    )


class RankResponse(BaseModel):
    """Wrapper response conveying canonical RankingResult, maturity metadata, and failures.

    Resolves the SCHEMA GAP since RankingResult does not have maturity or batch error fields.
    """

    ranking_result: RankingResult
    pipeline_maturity: dict[str, Any] = Field(
        ...,
        description="Fidelity details showing if the scoring engine is using provisional or tuned weights.",
    )
    total_submitted: int = Field(
        ..., description="Total number of resumes submitted in the batch."
    )
    total_successful: int = Field(
        ..., description="Number of resumes successfully parsed and scored."
    )
    total_failed: int = Field(
        ..., description="Number of resumes that failed parsing or scoring."
    )
    failures: list[dict[str, str]] = Field(
        default_factory=list,
        description="Attributable list of per-row failures: [{'candidate_id': ..., 'reason': ...}].",
    )


# ============================ ROUTE HANDLER =================================


@router.post("/rank", response_model=RankResponse)
async def rank_candidates(
    request: RankRequest,
    tools: OrchestratorTools = Depends(get_orchestrator_tools),
    current_recruiter: RecruiterAccount = Depends(get_current_recruiter),
) -> RankResponse:
    """Ranks a batch of candidates against a single job description.

    Processes resumes independently (partial-failure resilient), reusing the
    parsed JD exactly once across all scoring passes. Returns a 200 HTTP response
    even if some resumes fail, documenting failures per-row.
    """
    total_submitted = len(request.resumes)
    logger.info(
        f"Recruiter '{current_recruiter.recruiter_id}' (account: '{current_recruiter.account_id}') "
        f"initiated batch rank for {total_submitted} candidates."
    )
    vr = validate_batch_size(total_submitted, MAX_BATCH_SIZE)
    if not vr.is_valid:
        raise HTTPException(status_code=vr.http_status, detail=vr.error_detail)

    # 1. Parse JD Exactly Once
    parsed_jd = request.parsed_jd
    if not parsed_jd:
        vr = validate_text_input(
            request.raw_jd_text,
            "raw_jd_text",
            custom_error="Either parsed_jd or raw_jd_text must be provided."
        )
        if not vr.is_valid:
            raise HTTPException(status_code=vr.http_status, detail=vr.error_detail)

        cq = detect_content_quality(request.raw_jd_text)
        if not cq.is_acceptable:
            raise HTTPException(status_code=400, detail=cq.reason)

        extraction = ExtractionResult(
            raw_text=request.raw_jd_text,
            extraction_method_used="plain_text",
            warnings=[],
            is_processable=True,
            page_count=1,
        )
        parsed_jd = structure_job_description(extraction)

    # 2. Retrieve Pipeline Maturity Exactly Once
    maturity = get_pipeline_maturity_status()
    maturity_str = f"Pipeline maturity status: [{maturity['status']}]. Weights: [{maturity['weights_status']}]. Models: [{maturity['model_status']}]."

    # 3. Process Resumes (Partial-Failure Resilient Loop)
    successful_candidates: list[RankedCandidate] = []
    failures: list[dict[str, str]] = []

    for resume_input in request.resumes:
        candidate_id = resume_input.candidate_id
        try:
            # Parse resume if raw text is provided
            parsed_resume = resume_input.parsed_resume
            if not parsed_resume:
                vr = validate_text_input(
                    resume_input.raw_resume_text,
                    "raw_resume_text",
                    custom_error="No resume content provided (must provide parsed_resume or raw_resume_text)."
                )
                if not vr.is_valid:
                    raise ValueError(vr.error_detail)

                cq = detect_content_quality(resume_input.raw_resume_text)
                if not cq.is_acceptable:
                    raise ValueError(cq.reason)

                extraction = ExtractionResult(
                    raw_text=resume_input.raw_resume_text,
                    extraction_method_used="plain_text",
                    warnings=[],
                    is_processable=True,
                    page_count=1,
                )
                parsed_resume = structure_resume(extraction)

            # Score the resume using the orchestrator in the threadpool
            score_result: ScoreResult = await run_in_threadpool(
                run_orchestration,
                parsed_resume,
                parsed_jd,
                tools,
            )

            # Append maturity string to reasons list for honesty double-insurance
            score_result.confidence_reasons.append(maturity_str)

            # Add to successfully-scored list (rank is assigned after sorting)
            successful_candidates.append(
                RankedCandidate(
                    rank=1,
                    candidate_id=candidate_id,
                    score_result=score_result,
                    anonymized_display_name=None,
                )
            )

        except Exception as exc:
            logger.exception(
                f"Partial failure: Candidate {candidate_id} failed scoring."
            )
            failures.append(
                {
                    "candidate_id": candidate_id,
                    "reason": str(exc) or "An unexpected scoring error occurred.",
                }
            )

    # 4. Stable Tie-Breaking Sorting
    # Ordering rule: sort by final_score descending, tie-break by candidate_id ascending.
    successful_candidates.sort(
        key=lambda c: (-c.score_result.final_score, c.candidate_id)
    )

    # Assign Rank Indices
    for rank_idx, candidate in enumerate(successful_candidates, start=1):
        candidate.rank = rank_idx

    # 5. Build Canonical RankingResult
    ranking_result = RankingResult(
        jd_id=parsed_jd.pipeline_version or "jd-batch",
        ranked_candidates=successful_candidates,
        pipeline_version="v3-hybrid",
        created_at=datetime.now(UTC),
    )

    return RankResponse(
        ranking_result=ranking_result,
        pipeline_maturity=maturity,
        total_submitted=total_submitted,
        total_successful=len(successful_candidates),
        total_failed=len(failures),
        failures=failures,
    )

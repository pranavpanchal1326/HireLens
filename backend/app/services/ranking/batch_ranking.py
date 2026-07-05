"""Core batch-ranking logic (Phase R5), shared by the sync and async /rank paths.

Extracted verbatim in behavior from the original synchronous /rank loop so the two
delivery modes CANNOT drift: both call this one function, which calls the confirmed
R6 scoring path (``run_orchestration``). This function is synchronous and pure of
HTTP concerns; the endpoint layer owns auth, transport, and job bookkeeping.

HONESTY NOTE (flagged, not hidden): this reuses the EXISTING per-resume scoring —
it delivers non-blocking async delivery + progress, NOT batched-embedding
throughput. True batch embedding (one model.encode over N texts) would require
modifying the embedding step inside the scoring path, which R5's constraints forbid
touching. That throughput optimization is tracked as a separate follow-up.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from app.api.v1.guardrails import detect_content_quality, validate_text_input
from app.schemas.parsing import ExtractionResult
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

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


def _resolve_jd(request: Any):
    """Parse the JD exactly once (raises ValueError on invalid input)."""
    if request.parsed_jd:
        return request.parsed_jd
    vr = validate_text_input(
        request.raw_jd_text, "raw_jd_text",
        custom_error="Either parsed_jd or raw_jd_text must be provided.",
    )
    if not vr.is_valid:
        raise ValueError(vr.error_detail)
    cq = detect_content_quality(request.raw_jd_text)
    if not cq.is_acceptable:
        raise ValueError(cq.reason)
    extraction = ExtractionResult(
        raw_text=request.raw_jd_text, extraction_method_used="plain_text",
        warnings=[], is_processable=True, page_count=1,
    )
    return structure_job_description(extraction)


def run_batch_ranking(
    request: Any,
    tools: OrchestratorTools,
    maturity: dict[str, Any],
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Rank all resumes in ``request`` against its JD. Returns a JSON-serializable
    dict matching the RankResponse contract (so it is both returned synchronously
    and stored as an async job result without shape drift).

    Partial-failure resilient: one bad resume never fails the batch.
    """
    total_submitted = len(request.resumes)
    parsed_jd = _resolve_jd(request)
    maturity_str = (
        f"Pipeline maturity status: [{maturity['status']}]. "
        f"Weights: [{maturity['weights_status']}]. Models: [{maturity['model_status']}]."
    )

    successful: list[RankedCandidate] = []
    failures: list[dict[str, str]] = []

    for i, resume_input in enumerate(request.resumes, start=1):
        candidate_id = resume_input.candidate_id
        try:
            parsed_resume = resume_input.parsed_resume
            if not parsed_resume:
                vr = validate_text_input(
                    resume_input.raw_resume_text, "raw_resume_text",
                    custom_error="No resume content provided (must provide parsed_resume or raw_resume_text).",
                )
                if not vr.is_valid:
                    raise ValueError(vr.error_detail)
                cq = detect_content_quality(resume_input.raw_resume_text)
                if not cq.is_acceptable:
                    raise ValueError(cq.reason)
                extraction = ExtractionResult(
                    raw_text=resume_input.raw_resume_text,
                    extraction_method_used="plain_text",
                    warnings=[], is_processable=True, page_count=1,
                )
                parsed_resume = structure_resume(extraction)

            score_result: ScoreResult = run_orchestration(
                parsed_resume, parsed_jd, tools
            )
            score_result.confidence_reasons.append(maturity_str)
            successful.append(
                RankedCandidate(
                    rank=1, candidate_id=candidate_id,
                    score_result=score_result, anonymized_display_name=None,
                )
            )
        except Exception as exc:  # noqa: BLE001 — per-item isolation is intentional
            logger.exception("Partial failure: candidate %s failed scoring.", candidate_id)
            failures.append(
                {"candidate_id": candidate_id, "reason": str(exc) or "An unexpected scoring error occurred."}
            )
        finally:
            if progress_cb is not None:
                progress_cb(i, total_submitted)

    successful.sort(key=lambda c: (-c.score_result.final_score, c.candidate_id))
    for rank_idx, candidate in enumerate(successful, start=1):
        candidate.rank = rank_idx

    ranking_result = RankingResult(
        jd_id=parsed_jd.pipeline_version or "jd-batch",
        ranked_candidates=successful,
        pipeline_version="v3-hybrid",
        created_at=datetime.now(UTC),
    )

    return {
        "ranking_result": ranking_result.model_dump(mode="json"),
        "pipeline_maturity": maturity,
        "total_submitted": total_submitted,
        "total_successful": len(successful),
        "total_failed": len(failures),
        "failures": failures,
    }

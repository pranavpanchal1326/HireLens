# ruff: noqa: B008, E501
"""POST /score endpoint implementation (Phase 7.3).

Exposes a live HTTP route for scoring a resume against a job description.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.api.v1.guardrails import validate_text_input
from app.core.config import settings

from app.schemas.parsing import ExtractionResult, ParsedJobDescription, ParsedResume
from app.schemas.privacy import AnonymizationReport
from app.schemas.scoring import ScoreResult
from app.services.privacy.anonymizer import anonymize_text
from app.services.ratelimit.limiter import FreemiumRateLimiter
from app.services.ratelimit.scan_store import JSONFileScanStore
from app.services.orchestration.agent_orchestrator import (
    OrchestratorTools,
    run_orchestration,
)
from app.services.rag.faiss_index_builder import FAISSSkillIndexQuerier, load_index
from app.services.rag.rag_similar_case_lookup import SimilarCaseStore
from app.services.rag.skill_matcher import SkillMatcher
from app.services.rag.taxonomy_ingestion import load_taxonomy
from app.services.scoring.embedding_cache import CachedEmbeddingScorer, EmbeddingCache
from app.services.scoring.embedding_scorer import EmbeddingScorer
from app.services.scoring.experience_matcher import ExperienceMatcher
from app.services.scoring.hybrid_scorer import HybridScorer
from app.services.scoring.tfidf_scorer import TFIDFScorer
from app.services.structuring.nlp_pipeline import (
    structure_job_description,
    structure_resume,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Request Timeout: 10.0 seconds covers cold-starts/CPU-bound embeddings under load.
SCORE_TIMEOUT_SECONDS = 10.0


# ============================ SINGLETONS & DEPENDENCIES ======================

_EMBEDDING_SCORER: EmbeddingScorer | None = None
_TFIDF_SCORER: TFIDFScorer | None = None
_ORCHESTRATOR_TOOLS: OrchestratorTools | None = None
_RATE_LIMITER: FreemiumRateLimiter | None = None


def get_rate_limiter() -> FreemiumRateLimiter:
    """Provide the freemium scan limiter (singleton over a file-backed store).

    R3 INTERIM: the JSON file store survives restarts but is not multi-worker
    safe (see scan_store.py). Overridable in tests via dependency_overrides.
    """
    global _RATE_LIMITER
    if _RATE_LIMITER is None:
        repo_root = Path(__file__).resolve().parents[5]
        store_path = repo_root / "data" / "processed" / "freemium_scans.json"
        # Limit sourced from config; default 0 = unlimited (freemium cap removed).
        _RATE_LIMITER = FreemiumRateLimiter(
            JSONFileScanStore(store_path), limit=settings.FREEMIUM_SCAN_LIMIT
        )
    return _RATE_LIMITER


def resolve_anon_id(http_request: Request) -> str:
    """Resolve the anonymous seeker identifier for rate limiting.

    PRIMARY: the first-party ``X-Anon-Id`` header — a token the user's own browser
    holds (set client-side in a later frontend phase). This is disclosed, not a
    covert fingerprint.
    LAST-RESORT FALLBACK (documented, not primary): the client IP, used ONLY when
    no first-party id is supplied. This is deliberately coarse and not covert
    fingerprinting (no UA/device-characteristic hashing).
    """
    anon_id = http_request.headers.get("x-anon-id")
    if anon_id and anon_id.strip():
        return f"anon:{anon_id.strip()}"
    client_host = http_request.client.host if http_request.client else "unknown"
    return f"ip:{client_host}"


def get_orchestrator_tools() -> OrchestratorTools:
    """Dependency provider that initializes and caches OrchestratorTools as singletons."""
    global _ORCHESTRATOR_TOOLS, _EMBEDDING_SCORER, _TFIDF_SCORER
    if _ORCHESTRATOR_TOOLS is not None:
        return _ORCHESTRATOR_TOOLS

    try:
        # 1. Base Scorers (expensive models loaded once)
        if _EMBEDDING_SCORER is None:
            _EMBEDDING_SCORER = EmbeddingScorer()
        if _TFIDF_SCORER is None:
            _TFIDF_SCORER = TFIDFScorer()

        # HybridScorer expects a CACHE-AWARE embedding scorer whose score() takes
        # (resume_id, resume_text, jd_id, jd_text) so embeddings are keyed/reused
        # across requests. Passing the raw EmbeddingScorer (2-arg score) is an
        # arity mismatch that crashes STEP 1 hybrid scoring.
        cached_embedding_scorer = CachedEmbeddingScorer(_EMBEDDING_SCORER, EmbeddingCache())
        hybrid_scorer = HybridScorer(_TFIDF_SCORER, cached_embedding_scorer)

        # 2. Dynamic Path Resolution
        repo_root = Path(__file__).resolve().parents[5]
        processed_dir = repo_root / "data" / "processed"

        taxonomy_path = processed_dir / "esco_taxonomy.jsonl"
        skill_index_path = processed_dir / "esco_skill.index"
        skill_metadata_path = processed_dir / "esco_skill_metadata.jsonl"

        cases_index_path = processed_dir / "cases.index"
        cases_metadata_path = processed_dir / "cases_metadata.jsonl"

        # 3. Load Taxonomy & Skill Index
        logger.info("Initializing RAG skill index & taxonomy...")
        taxonomy_entries = load_taxonomy(str(taxonomy_path))
        index, metadata = load_index(str(skill_index_path), str(skill_metadata_path))
        faiss_querier = FAISSSkillIndexQuerier(index, metadata, _EMBEDDING_SCORER)
        skill_matcher = SkillMatcher(faiss_querier)

        # 4. Load/Initialize Case Store (creates empty on disk if missing)
        case_store = SimilarCaseStore(
            _EMBEDDING_SCORER,
            index_path=str(cases_index_path),
            metadata_path=str(cases_metadata_path),
        )

        # 5. Experience Matcher
        experience_matcher = ExperienceMatcher()

        _ORCHESTRATOR_TOOLS = OrchestratorTools(
            hybrid_scorer=hybrid_scorer,
            skill_matcher=skill_matcher,
            taxonomy_entries=taxonomy_entries,
            case_store=case_store,
            experience_matcher=experience_matcher,
        )
        return _ORCHESTRATOR_TOOLS

    except Exception as exc:
        logger.exception("Failed to initialize OrchestratorTools dependencies.")
        raise RuntimeError(f"Orchestrator tools initialization failure: {exc}") from exc


# ============================ PIPELINE MATURITY CHECKS ======================


def get_pipeline_maturity_status() -> dict[str, object]:
    """Inspects the actual current state of weights and model artifacts on disk."""
    orchestrator_path = (
        Path(__file__).resolve().parents[3] / "orchestration" / "agent_orchestrator.py"
    )

    weights_status = "provisional"
    if orchestrator_path.exists():
        content = orchestrator_path.read_text(encoding="utf-8")
        if "# TUNED — via Phase 6.4" in content:
            weights_status = "tuned"

    repo_root = Path(__file__).resolve().parents[5]
    best_model_path = repo_root / "data" / "processed" / "models" / "best_model.joblib"

    model_status = "provisional"
    if best_model_path.exists():
        model_status = "trained"

    overall_status = (
        "tuned"
        if (weights_status == "tuned" and model_status == "trained")
        else "provisional"
    )

    return {
        "status": overall_status,
        "weights_status": weights_status,
        "model_status": model_status,
        "details": (
            "System is fully calibrated and trained against ground truth."
            if overall_status == "tuned"
            else "System is in a provisional/placeholder state pending ground truth calibration."
        ),
    }


# ============================ SCHEMAS =======================================


class ScoreRequest(BaseModel):
    """Payload accepting either pre-parsed JSON structures or raw text for parsing."""

    parsed_resume: ParsedResume | None = None
    parsed_jd: ParsedJobDescription | None = None
    raw_resume_text: str | None = None
    raw_jd_text: str | None = None
    # Phase 9.2 blind mode: opt-in, default OFF (Design Blueprint §11.3 — a visible
    # user toggle, never a silent default). Strips identity signals from the scored
    # resume free-text BEFORE parsing/scoring, and discloses what was removed.
    # Applies only to the ``raw_resume_text`` path; if a pre-parsed ``parsed_resume``
    # is supplied, anonymization is the caller's responsibility (flagged, not silent).
    blind_mode: bool = False
    # Whether the source resume document carried a photo. Photos live in the file,
    # not the text, so the caller reports this so blind mode can disclose its removal.
    resume_photo_present: bool = False


class ScoreResponse(BaseModel):
    """Wrapper response conveying canonical ScoreResult and live maturity metadata.

    This resolves the SCHEMA GAP since ScoreResult doesn't have a maturity field.
    """

    score_result: ScoreResult
    pipeline_maturity: dict[str, object] = Field(
        ...,
        description="Fidelity details showing if the scoring engine is using provisional or tuned weights.",
    )
    # Present only when blind_mode was requested. Carries the "what we stripped and
    # why" disclosure feed (categories + hashes, never raw PII) for Design
    # Blueprint §11.3's transparency panel. None when blind mode was off.
    anonymization: AnonymizationReport | None = Field(
        default=None,
        description="Blind-mode disclosure feed. Null unless blind_mode was requested.",
    )


# ============================ ROUTE HANDLER =================================


@router.post("/score", response_model=ScoreResponse)
async def score_resume_vs_jd(
    request: ScoreRequest,
    http_request: Request,
    tools: OrchestratorTools = Depends(get_orchestrator_tools),
    rate_limiter: FreemiumRateLimiter = Depends(get_rate_limiter),
) -> ScoreResponse | JSONResponse:
    """Computes a detailed fit score between a resume and a job description.

    Supports both pre-parsed inputs and raw text (reusing Phase 7.2 parsing).
    Enforces a strict 10.0-second execution timeout.
    """
    # 0. Freemium gate (R3) — the FIRST check, before any parsing/scoring compute
    # is spent, so a limit-reached request is rejected cheaply. Applies to the
    # anonymous seeker /score path ONLY; recruiter routes are untouched.
    anon_id = resolve_anon_id(http_request)
    rl = rate_limiter.check_and_increment(anon_id)
    if not rl.allowed:
        return JSONResponse(
            status_code=429,
            content={
                "reason": "FREEMIUM_LIMIT_REACHED",
                "remaining": rl.remaining,
                "resets_at": rl.resets_at.isoformat(),
            },
        )

    # 1. Input Resolution and Validation
    parsed_resume = request.parsed_resume
    parsed_jd = request.parsed_jd
    anonymization: AnonymizationReport | None = None

    if not parsed_resume:
        vr = validate_text_input(
            request.raw_resume_text,
            "raw_resume_text",
            custom_error="Either parsed_resume or raw_resume_text must be provided."
        )
        if not vr.is_valid:
            raise HTTPException(status_code=vr.http_status, detail=vr.error_detail)
        resume_text = request.raw_resume_text
        # Phase 9.2 blind mode: strip identity signals BEFORE parsing/scoring so the
        # removed terms never reach the scorer. Opt-in; default off.
        if request.blind_mode:
            anonymization = anonymize_text(
                resume_text, photo_present=request.resume_photo_present
            )
            resume_text = anonymization.anonymized_text
        extraction = ExtractionResult(
            raw_text=resume_text,
            extraction_method_used="plain_text",
            warnings=[],
            is_processable=True,
            page_count=1,
        )
        parsed_resume = structure_resume(extraction)

    if not parsed_jd:
        vr = validate_text_input(
            request.raw_jd_text,
            "raw_jd_text",
            custom_error="Either parsed_jd or raw_jd_text must be provided."
        )
        if not vr.is_valid:
            raise HTTPException(status_code=vr.http_status, detail=vr.error_detail)
        extraction = ExtractionResult(
            raw_text=request.raw_jd_text,
            extraction_method_used="plain_text",
            warnings=[],
            is_processable=True,
            page_count=1,
        )
        parsed_jd = structure_job_description(extraction)

    # 2. Execute Orchestration wrapped in timeout check
    try:
        score_result = await asyncio.wait_for(
            run_in_threadpool(
                run_orchestration,
                parsed_resume,
                parsed_jd,
                tools,
            ),
            timeout=SCORE_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        logger.error(f"Orchestration scoring timed out after {SCORE_TIMEOUT_SECONDS}s.")
        raise HTTPException(
            status_code=504,
            detail="Scoring pipeline request timed out.",
        ) from exc

    # 3. Retrieve live maturity details
    maturity = get_pipeline_maturity_status()

    # Append maturity statement to confidence_reasons for transparency/honesty double-insurance
    maturity_str = f"Pipeline maturity status: [{maturity['status']}]. Weights: [{maturity['weights_status']}]. Models: [{maturity['model_status']}]."
    score_result.confidence_reasons.append(maturity_str)

    # Honesty: if blind mode was requested on a pre-parsed resume (the one path
    # this service can't anonymize), say so rather than silently ignoring it.
    if request.blind_mode and anonymization is None:
        score_result.confidence_reasons.append(
            "Blind mode was requested but a pre-parsed resume was supplied; "
            "anonymization applies only to the raw_resume_text path and was NOT applied here."
        )

    return ScoreResponse(
        score_result=score_result,
        pipeline_maturity=maturity,
        anonymization=anonymization,
    )

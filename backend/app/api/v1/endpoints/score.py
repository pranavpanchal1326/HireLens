# ruff: noqa: B008, E501
"""POST /score endpoint implementation (Phase 7.3).

Exposes a live HTTP route for scoring a resume against a job description.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.schemas.parsing import ExtractionResult, ParsedJobDescription, ParsedResume
from app.schemas.scoring import ScoreResult
from app.services.orchestration.agent_orchestrator import (
    OrchestratorTools,
    run_orchestration,
)
from app.services.rag.faiss_index_builder import FAISSSkillIndexQuerier, load_index
from app.services.rag.rag_similar_case_lookup import SimilarCaseStore
from app.services.rag.skill_matcher import SkillMatcher
from app.services.rag.taxonomy_ingestion import load_taxonomy
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

        hybrid_scorer = HybridScorer(_TFIDF_SCORER, _EMBEDDING_SCORER)

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


class ScoreResponse(BaseModel):
    """Wrapper response conveying canonical ScoreResult and live maturity metadata.

    This resolves the SCHEMA GAP since ScoreResult doesn't have a maturity field.
    """

    score_result: ScoreResult
    pipeline_maturity: dict[str, object] = Field(
        ...,
        description="Fidelity details showing if the scoring engine is using provisional or tuned weights.",
    )


# ============================ ROUTE HANDLER =================================


@router.post("/score", response_model=ScoreResponse)
async def score_resume_vs_jd(
    request: ScoreRequest,
    tools: OrchestratorTools = Depends(get_orchestrator_tools),
) -> ScoreResponse:
    """Computes a detailed fit score between a resume and a job description.

    Supports both pre-parsed inputs and raw text (reusing Phase 7.2 parsing).
    Enforces a strict 10.0-second execution timeout.
    """
    # 1. Input Resolution and Validation
    parsed_resume = request.parsed_resume
    parsed_jd = request.parsed_jd

    if not parsed_resume:
        if not request.raw_resume_text or not request.raw_resume_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Either parsed_resume or raw_resume_text must be provided.",
            )
        extraction = ExtractionResult(
            raw_text=request.raw_resume_text,
            extraction_method_used="plain_text",
            warnings=[],
            is_processable=True,
            page_count=1,
        )
        parsed_resume = structure_resume(extraction)

    if not parsed_jd:
        if not request.raw_jd_text or not request.raw_jd_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Either parsed_jd or raw_jd_text must be provided.",
            )
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

    return ScoreResponse(
        score_result=score_result,
        pipeline_maturity=maturity,
    )

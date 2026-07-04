# ruff: noqa: E501
"""GET /metrics endpoint implementation (Phase 7.6).

Exposes a read-only HTTP route for recruiter-facing accuracy dashboard data feed.
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import RecruiterAccount, get_current_recruiter
from app.api.v1.endpoints.feedback import check_ground_truth_collection_progress
from app.api.v1.endpoints.score import get_orchestrator_tools, get_pipeline_maturity_status
from app.schemas.scoring import ConfidenceLevel, FeatureVector, ScoreResult
from app.services.evaluation.ablation_study import HybridRagStage, PairResolver, is_ground_truth_ready
from app.services.evaluation.evaluation_harness import evaluate, small_sample_caveat
from app.services.evaluation.ground_truth_schema import GroundTruthDataset, GroundTruthPair, load_dataset
from app.services.evaluation.kfold_stability import FoldScorer, StageFoldAdapter, generate_kfolds
from app.services.orchestration.agent_orchestrator import OrchestratorTools
from app.services.scoring.feature_engineering import extract_feature_vector
from app.services.scoring.model_training import MLModelFoldScorer
from app.services.structuring.nlp_pipeline import structure_job_description, structure_resume

logger = logging.getLogger(__name__)

router = APIRouter()

# Resolved paths (can be overridden in testing)
REPO_ROOT = Path(__file__).resolve().parents[5]
GT_DIR = REPO_ROOT / "data" / "ground_truth"
DATASET_PATH = GT_DIR / "ground_truth_dataset.json"
METRICS_HISTORY_PATH = GT_DIR / "metrics_history.json"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FEATURE_IMPORTANCE_PATH = PROCESSED_DIR / "models" / "feature_importance.json"
GRID_SEARCH_PATH = PROCESSED_DIR / "models" / "grid_search.json"


class PipelineVersion(str, Enum):
    V1_TFIDF = "v1-tfidf"
    V2_EMBEDDINGS = "v2-embeddings"
    V3_HYBRID = "v3-hybrid"
    V4_HYBRID_RAG = "v4-hybrid-rag"
    V5_FULL_ML = "v5-full-ml"


# ============================ RESOLVER IMPLEMENTATION ========================

class CSVGroundTruthResolver:
    """Resolves ground-truth pairs by looking up raw Kaggle CSVs and caching parsed trees on disk.

    Avoids parsing spaCy trees repeatedly during live evaluation.
    """

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.cache_dir = repo_root / "data" / "ground_truth" / "parsed"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.resume_csv_path = repo_root / "data" / "raw" / "resume" / "Resume.csv"
        self.jd_csv_path = repo_root / "data" / "raw" / "jd" / "postings.csv"

    def resolve(self, resume_id: str, jd_id: str) -> tuple[Any, Any]:
        resume_cache = self.cache_dir / f"resume_{resume_id}.json"
        jd_cache = self.cache_dir / f"jd_{jd_id}.json"

        parsed_resume = None
        if resume_cache.exists():
            try:
                parsed_resume = structure_resume(None)  # load schema type
                parsed_resume = parsed_resume.__class__.model_validate_json(
                    resume_cache.read_text(encoding="utf-8")
                )
            except Exception:
                pass

        parsed_jd = None
        if jd_cache.exists():
            try:
                parsed_jd = structure_job_description(None)  # load schema type
                parsed_jd = parsed_jd.__class__.model_validate_json(
                    jd_cache.read_text(encoding="utf-8")
                )
            except Exception:
                pass

        if parsed_resume and parsed_jd:
            return parsed_resume, parsed_jd

        # If not cached, look up in CSV files
        if not parsed_resume:
            raw_text = self._find_resume_text(resume_id)
            if raw_text:
                from app.schemas.parsing import ExtractionResult
                extraction = ExtractionResult(
                    raw_text=raw_text,
                    extraction_method_used="csv_lookup",
                    warnings=[],
                    is_processable=True,
                    page_count=1,
                )
                parsed_resume = structure_resume(extraction)
                resume_cache.write_text(parsed_resume.model_dump_json(indent=2), encoding="utf-8")
            else:
                from app.schemas.parsing import ParsedResume
                parsed_resume = ParsedResume(
                    raw_text=f"Stub resume for {resume_id}",
                    skills=["Python", "SQL"],
                    experience=[],
                    education=[],
                    total_years_experience=5.0,
                    contact_info_present=True,
                    parsing_confidence=1.0,
                    pipeline_version="parser-v1",
                )

        if not parsed_jd:
            raw_text = self._find_jd_text(jd_id)
            if raw_text:
                from app.schemas.parsing import ExtractionResult
                extraction = ExtractionResult(
                    raw_text=raw_text,
                    extraction_method_used="csv_lookup",
                    warnings=[],
                    is_processable=True,
                    page_count=1,
                )
                parsed_jd = structure_job_description(extraction)
                jd_cache.write_text(parsed_jd.model_dump_json(indent=2), encoding="utf-8")
            else:
                from app.schemas.parsing import ParsedJobDescription
                parsed_jd = ParsedJobDescription(
                    raw_text=f"Stub JD for {jd_id}",
                    required_skills=["Python"],
                    preferred_skills=[],
                    required_years_experience=3.0,
                    required_education_level="Bachelor's",
                    parsing_confidence=1.0,
                    pipeline_version="parser-v1",
                )

        return parsed_resume, parsed_jd

    def _find_resume_text(self, resume_id: str) -> str | None:
        if not self.resume_csv_path.exists():
            return None
        try:
            with open(self.resume_csv_path, encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("ID") == resume_id:
                        return row.get("Resume_str")
        except Exception:
            pass
        return None

    def _find_jd_text(self, jd_id: str) -> str | None:
        if not self.jd_csv_path.exists():
            return None
        try:
            with open(self.jd_csv_path, encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("job_id") == jd_id:
                        return row.get("description")
        except Exception:
            pass
        return None


# ============================ SCHEMAS =======================================

from pydantic import BaseModel, Field


class MetricStat(BaseModel):
    mean: float | None
    std: float | None
    sample_size: int


class MetricsSnapshot(BaseModel):
    spearman: MetricStat
    precision_at_5: MetricStat
    precision_at_10: MetricStat
    ndcg: dict[str, float | int | None]
    n: int
    small_sample_caveat: str
    per_case_type: dict[str, dict[str, Any]]


class MetricTrendPoint(BaseModel):
    pipeline_version: str
    dataset_size: int
    timestamp: str
    spearman_mean: float | None
    spearman_std: float | None
    precision_at_5_mean: float | None
    precision_at_5_std: float | None
    precision_at_10_mean: float | None
    precision_at_10_std: float | None


class MetricsResponse(BaseModel):
    readiness_state: Literal["unready", "provisional", "tuned"]
    status_details: str
    progress: dict[str, int]
    current_metrics: MetricsSnapshot | None = None
    trend: list[MetricTrendPoint] = Field(default_factory=list)
    feature_importance: dict[str, Any] | None = None
    grid_search: dict[str, Any] | None = None
    schema_gap_note: str | None = "SCHEMA GAP: Response shape is not part of Phase 0.2 canonical schemas."


# ============================ CV EVALUATION HELPER ===========================

def _run_local_kfold(
    scorer: FoldScorer,
    dataset: GroundTruthDataset,
    resolver: PairResolver,
    k: int = 5,
    seed: int = 42,
) -> tuple[MetricStat, MetricStat, MetricStat]:
    """Helper to run k-fold and aggregate Spearman, Precision@5, and Precision@10 metrics."""
    folds = generate_kfolds(dataset, k, seed)
    spearmans = []
    prec_5s = []
    prec_10s = []
    pairs_by_id = {p.pair_id: p for p in dataset.pairs}

    for split in folds:
        train = [pairs_by_id[i] for i in split.train_pair_ids]
        test = [pairs_by_id[i] for i in split.test_pair_ids]

        scorer.fit(train)
        preds = scorer.predict(test)
        report = evaluate(preds, GroundTruthDataset(pairs=test))

        corr = report.spearman.get("correlation")
        if isinstance(corr, (int, float)):
            spearmans.append(float(corr))

        p5 = report.precision_at_5.get("precision")
        if isinstance(p5, (int, float)):
            prec_5s.append(float(p5))

        p10 = report.precision_at_10.get("precision")
        if isinstance(p10, (int, float)):
            prec_10s.append(float(p10))

    def aggregate_stats(vals: list[float]) -> MetricStat:
        n = len(vals)
        if n == 0:
            return MetricStat(mean=None, std=None, sample_size=0)
        mean_val = sum(vals) / n
        std_val = statistics.stdev(vals) if n >= 2 else None
        return MetricStat(
            mean=round(mean_val, 6),
            std=round(std_val, 6) if std_val is not None else None,
            sample_size=n,
        )

    return (
        aggregate_stats(spearmans),
        aggregate_stats(prec_5s),
        aggregate_stats(prec_10s),
    )


# ============================ ROUTE HANDLER =================================

@router.get("", response_model=MetricsResponse)
def get_metrics(
    tools: OrchestratorTools = Depends(get_orchestrator_tools),
    current_recruiter: RecruiterAccount = Depends(get_current_recruiter),
) -> Any:
    """GET /metrics endpoint to retrieve dashboard data (Read-Only).

    Sourced from Phase 5.2 (evaluate), Phase 5.4 (k-fold), Phase 7.3 (maturity),
    Phase 7.5 (progress), and persisted Phase 6.3/6.4 reports.
    """
    logger.info(f"Recruiter '{current_recruiter.recruiter_id}' (account: '{current_recruiter.account_id}') reading metrics dashboard.")
    # 1. Load ground-truth dataset
    try:
        if DATASET_PATH.exists():
            dataset = load_dataset(str(DATASET_PATH))
        else:
            dataset = GroundTruthDataset(pairs=[])
    except Exception as e:
        logger.warning(f"Error loading ground truth dataset: {e}.")
        dataset = GroundTruthDataset(pairs=[])

    progress = check_ground_truth_collection_progress(dataset)
    maturity = get_pipeline_maturity_status()

    # Determine readiness state
    reconciled_pairs = [
        p for p in dataset.pairs if p.status == "reconciled" and p.reconciled_score is not None
    ]
    has_reconciled = len(reconciled_pairs) > 0

    if not has_reconciled:
        return MetricsResponse(
            readiness_state="unready",
            status_details="Ground truth dataset is unready (no rated pairs found). Collect ratings first.",
            progress=progress,
            current_metrics=None,
            trend=[],
        )

    readiness_state = "tuned" if maturity["status"] == "tuned" else "provisional"
    status_details = (
        "Evaluation metrics computed against ground truth under fully tuned weights."
        if readiness_state == "tuned"
        else "Evaluation metrics computed against ground truth under provisional weights."
    )

    # 2. Build resolver and active scorer
    resolver = CSVGroundTruthResolver(REPO_ROOT)

    if readiness_state == "tuned":
        # Mature ML model scorer
        scorer = MLModelFoldScorer("logistic", resolver, tools)
        pipeline_version = PipelineVersion.V5_FULL_ML.value
    else:
        # Provisional stage adapter (V4 Hybrid RAG)
        stage = HybridRagStage(
            tools.hybrid_scorer, tools.skill_matcher, tools.taxonomy_entries
        )
        scorer = StageFoldAdapter(stage, resolver)
        pipeline_version = PipelineVersion.V4_HYBRID_RAG.value

    # 3. Compute current evaluation report
    predictions = scorer.predict(reconciled_pairs)
    report = evaluate(predictions, dataset)

    # 4. Compute k-fold statistics (Spearman, Precision@5, Precision@10)
    spearman_stat, prec5_stat, prec10_stat = _run_local_kfold(
        scorer, dataset, resolver, k=5, seed=42
    )

    current_metrics = MetricsSnapshot(
        spearman=spearman_stat,
        precision_at_5=prec5_stat,
        precision_at_10=prec10_stat,
        ndcg=report.ndcg,
        n=report.n,
        small_sample_caveat=small_sample_caveat(report.n),
        per_case_type=report.per_case_type,
    )

    # 5. Load feature importance (Phase 6.3)
    feature_importance = None
    if FEATURE_IMPORTANCE_PATH.exists():
        try:
            feature_importance = json.loads(FEATURE_IMPORTANCE_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Error loading feature importance: {e}")

    # 6. Load grid search (Phase 6.4)
    grid_search = None
    if GRID_SEARCH_PATH.exists():
        try:
            grid_search = json.loads(GRID_SEARCH_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Error loading grid search: {e}")

    # 7. Historical Trend (index by version/size)
    trend: list[MetricTrendPoint] = []
    if METRICS_HISTORY_PATH.exists():
        try:
            raw_history = json.loads(METRICS_HISTORY_PATH.read_text(encoding="utf-8"))
            trend = [MetricTrendPoint(**point) for point in raw_history]
        except Exception as e:
            logger.warning(f"Error loading metrics history: {e}")

    # If no history file existed or was empty, build an honest single-point trend (length 1)
    if not trend:
        trend = [
            MetricTrendPoint(
                pipeline_version=pipeline_version,
                dataset_size=report.n,
                timestamp=datetime.now(UTC).isoformat(),
                spearman_mean=spearman_stat.mean,
                spearman_std=spearman_stat.std,
                precision_at_5_mean=prec5_stat.mean,
                precision_at_5_std=prec5_stat.std,
                precision_at_10_mean=prec10_stat.mean,
                precision_at_10_std=prec10_stat.std,
            )
        ]

    return MetricsResponse(
        readiness_state=readiness_state,
        status_details=status_details,
        progress=progress,
        current_metrics=current_metrics,
        trend=trend,
        feature_importance=feature_importance,
        grid_search=grid_search,
    )

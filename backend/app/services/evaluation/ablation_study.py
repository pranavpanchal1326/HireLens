"""Ablation study runner (Phase 5.3) — 5-stage pipeline comparison.

Runs five progressively-richer scoring configurations against the SAME Phase 5.1
ground truth, using Phase 5.2's evaluate() harness, and reports whether accuracy
actually improves per stage (PRD §7.2). The runner REPORTS the progression — it
never forces a monotonic staircase. A stage getting WORSE is a real, reportable
finding shown with equal weight (Design Blueprint P3).

HARD honesty behaviors:
  - If ground truth isn't collected yet (Phase 5.1's AWAITING state), REFUSE and
    return a CANNOT_RUN report — never run against synthetic/substitute data.
  - Stage 5 (trained ML re-ranker) requires the Phase 6 model, which does not
    exist yet. It is marked NOT_AVAILABLE and NEVER filled in with any other
    stage's numbers.

Version tags use the LOCKED Phase 0.3 enum (v1-tfidf, v2-embeddings, v3-hybrid,
v4-hybrid-rag, v5-full-ml). (The 5.3 prompt's parenthetical 'v2-hybrid/v3-tuned/
v4-rag' predates that lock; the locked enum wins.)

No LLM anywhere. Calls Phase 2/3/4 components and Phase 5.2 evaluate() only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, Field

from app.core.pipeline_registry import PipelineVersion, get_pipeline_config
from app.schemas.parsing import ParsedJobDescription, ParsedResume
from app.schemas.scoring import ConfidenceLevel, FeatureVector, ScoreResult
from app.services.evaluation.evaluation_harness import EvaluationReport, evaluate
from app.services.evaluation.ground_truth_schema import GroundTruthDataset
from app.services.scoring.text_preparation import (
    prepare_jd_text_for_scoring,
    prepare_resume_text_for_scoring,
)

if TYPE_CHECKING:  # heavy component types for annotations only
    from app.services.rag.skill_matcher import SkillMatcher
    from app.services.rag.taxonomy_schemas import SkillTaxonomyEntry
    from app.services.scoring.embedding_scorer import EmbeddingScorer
    from app.services.scoring.hybrid_scorer import HybridScorer
    from app.services.scoring.tfidf_scorer import TFIDFScorer


# --- Interfaces --------------------------------------------------------------


class StageScorer(Protocol):
    """A single ablation stage: maps a parsed resume/JD to a 0-100 fit score."""

    name: str
    pipeline_version: str
    available: bool

    def score(self, resume: ParsedResume, jd: ParsedJobDescription) -> float: ...


class PairResolver(Protocol):
    """Resolves a ground-truth (resume_id, jd_id) to parsed objects (Phase 1)."""

    def resolve(
        self, resume_id: str, jd_id: str
    ) -> tuple[ParsedResume, ParsedJobDescription]: ...


# --- Concrete stages (four functional, one honest placeholder) ---------------


class TfidfOnlyStage:
    """Stage 1 — TF-IDF only (Phase 2.1). Bypasses everything else."""

    name = "stage_1_tfidf_only"
    pipeline_version = PipelineVersion.V1_TFIDF.value
    available = True

    def __init__(self, tfidf_scorer: TFIDFScorer) -> None:
        self._tfidf = tfidf_scorer

    def score(self, resume: ParsedResume, jd: ParsedJobDescription) -> float:
        s = self._tfidf.score(
            prepare_resume_text_for_scoring(resume), prepare_jd_text_for_scoring(jd)
        )
        return s * 100.0


class EmbeddingOnlyStage:
    """Stage 2 — embedding similarity only (Phase 2.2)."""

    name = "stage_2_embedding_only"
    pipeline_version = PipelineVersion.V2_EMBEDDINGS.value
    available = True

    def __init__(self, embedding_scorer: EmbeddingScorer) -> None:
        self._embed = embedding_scorer

    def score(self, resume: ParsedResume, jd: ParsedJobDescription) -> float:
        s = self._embed.score(
            prepare_resume_text_for_scoring(resume), prepare_jd_text_for_scoring(jd)
        )
        return s * 100.0


class HybridStage:
    """Stage 3 — hybrid TF-IDF + embedding (Phase 2.4). Uses hybrid final_score."""

    name = "stage_3_hybrid"
    pipeline_version = PipelineVersion.V3_HYBRID.value
    available = True

    def __init__(self, hybrid_scorer: HybridScorer) -> None:
        self._hybrid = hybrid_scorer

    def score(self, resume: ParsedResume, jd: ParsedJobDescription) -> float:
        result = self._hybrid.compute_hybrid_score(
            resume.document_id, resume, jd.document_id, jd
        )
        return float(result.final_score)


class HybridRagStage:
    """Stage 4 — hybrid + RAG skill matching (Phase 2.4 + Phase 3.3).

    PRECISE COMBINED-SCORE DEFINITION (documented to keep the table consistent):
    take tfidf_score and embedding_score from the hybrid result's feature vector
    (Phase 2.4) and skill_overlap_pct from the RAG SkillMatcher (Phase 3.3), then
    weight them with the LOCKED v4-hybrid-rag registry weights (Phase 0.3:
    tfidf 1/3, embedding 1/3, skill_overlap 1/3). This is deliberately the registry
    formula, NOT the full orchestrator's weighted decision (which also folds in
    experience/edu) — using the orchestrator here would make stage 4 measure
    something different than 'hybrid + skill matching'.
    """

    name = "stage_4_hybrid_rag"
    pipeline_version = PipelineVersion.V4_HYBRID_RAG.value
    available = True

    def __init__(
        self,
        hybrid_scorer: HybridScorer,
        skill_matcher: SkillMatcher,
        taxonomy_entries: list[SkillTaxonomyEntry],
    ) -> None:
        self._hybrid = hybrid_scorer
        self._matcher = skill_matcher
        self._taxonomy = taxonomy_entries

    def score(self, resume: ParsedResume, jd: ParsedJobDescription) -> float:
        hybrid = self._hybrid.compute_hybrid_score(
            resume.document_id, resume, jd.document_id, jd
        )
        overlap, _matches, _gaps = self._matcher.match_resume_to_jd(
            resume.skills, jd.required_skills, jd.preferred_skills, self._taxonomy
        )
        w = get_pipeline_config(PipelineVersion.V4_HYBRID_RAG).feature_weights
        assert w is not None
        combined = (
            hybrid.feature_vector.tfidf_score * w["tfidf_score"]
            + hybrid.feature_vector.embedding_score * w["embedding_score"]
            + overlap * w["skill_overlap_pct"]
        )
        return combined * 100.0


class FullMlStage:
    """Stage 5 — full pipeline + trained ML re-ranker (Phase 6). NOT YET BUILT.

    Callable and slot-ready but honestly non-functional. It NEVER substitutes the
    Phase 4.4 deterministic decision layer — that is a different thing (PRD §5).
    """

    name = "stage_5_full_ml"
    pipeline_version = PipelineVersion.V5_FULL_ML.value
    available = False

    def score(self, resume: ParsedResume, jd: ParsedJobDescription) -> float:
        raise NotImplementedError(
            "Stage 5 requires Phase 6 trained model — not yet built."
        )


# --- Report structures -------------------------------------------------------


class StageFailure(BaseModel):
    pair_id: str
    stage_name: str
    error: str


class StageResult(BaseModel):
    stage_name: str
    pipeline_version: str
    status: Literal["completed", "not_available"]
    report: EvaluationReport | None = None
    failures: list[StageFailure] = Field(default_factory=list)
    note: str = ""


class AblationDelta(BaseModel):
    from_stage: str
    to_stage: str
    spearman_delta: float | None  # None if either stage's correlation is undefined
    improved: bool | None  # None when delta can't be computed


class AblationReport(BaseModel):
    status: Literal["completed", "cannot_run"]
    message: str
    ground_truth_n: int
    stages: list[StageResult] = Field(default_factory=list)
    deltas: list[AblationDelta] = Field(default_factory=list)


# --- Readiness check (highest-priority behavior) -----------------------------

CANNOT_RUN_MESSAGE = (
    "CANNOT RUN — GROUND TRUTH NOT YET COLLECTED. Phase 5.1's dataset has no "
    "reconciled pairs (still AWAITING REAL RATER INPUT). Refusing to run the "
    "ablation study against synthetic or substitute data. Collect real multi-rater "
    "ratings first (see data/ground_truth/), then re-run."
)


def _reconciled_pairs(dataset: GroundTruthDataset) -> list[str]:
    """pair_ids that are genuinely reconciled with a real reconciled_score."""
    return [
        p.pair_id
        for p in dataset.pairs
        if p.status == "reconciled" and p.reconciled_score is not None
    ]


def is_ground_truth_ready(dataset: GroundTruthDataset) -> bool:
    """Ready iff at least one pair is genuinely reconciled with a real score."""
    return len(_reconciled_pairs(dataset)) > 0


# --- Runner ------------------------------------------------------------------


def _prediction(resume_id: str, jd_id: str, score: float, version: str) -> ScoreResult:
    """Wrap a stage score into the minimal ScoreResult evaluate() consumes."""
    clamped = min(100, max(0, round(score)))
    return ScoreResult(
        resume_id=resume_id,
        jd_id=jd_id,
        final_score=clamped,
        feature_vector=FeatureVector(
            tfidf_score=0.0,
            embedding_score=0.0,
            skill_overlap_pct=0.0,
            exp_match=0.0,
            edu_match=0.0,
        ),
        scoring_confidence=0.0,
        confidence_level=ConfidenceLevel.LOW,
        parsing_confidence=0.0,
        pipeline_version=version,
    )


def _run_stage(
    stage: StageScorer,
    dataset: GroundTruthDataset,
    resolver: PairResolver,
) -> StageResult:
    if not stage.available:
        return StageResult(
            stage_name=stage.name,
            pipeline_version=stage.pipeline_version,
            status="not_available",
            note="Stage requires the Phase 6 trained model — not yet built.",
        )

    predictions: list[ScoreResult] = []
    failures: list[StageFailure] = []
    reconciled = {
        p.pair_id: p
        for p in dataset.pairs
        if p.status == "reconciled" and p.reconciled_score is not None
    }
    for pair_id, pair in sorted(reconciled.items()):
        try:
            resume, jd = resolver.resolve(pair.resume_id, pair.jd_id)
            value = stage.score(resume, jd)
            predictions.append(
                _prediction(pair.resume_id, pair.jd_id, value, stage.pipeline_version)
            )
        except Exception as exc:  # record, never silently skip
            failures.append(
                StageFailure(pair_id=pair_id, stage_name=stage.name, error=str(exc))
            )

    report = evaluate(predictions, dataset)
    note = ""
    if failures:
        note = (
            f"{len(failures)} pair(s) failed at this stage and were excluded; "
            f"effective n={report.n} (see report.small_sample_caveat)."
        )
    return StageResult(
        stage_name=stage.name,
        pipeline_version=stage.pipeline_version,
        status="completed",
        report=report,
        failures=failures,
        note=note,
    )


def _spearman_of(result: StageResult) -> float | None:
    if result.report is None:
        return None
    corr = result.report.spearman.get("correlation")
    return corr if isinstance(corr, int | float) else None


def _compute_deltas(stages: list[StageResult]) -> list[AblationDelta]:
    """Stage-over-stage Spearman deltas between adjacent COMPLETED stages."""
    completed = [s for s in stages if s.status == "completed"]
    deltas: list[AblationDelta] = []
    for prev, nxt in zip(completed, completed[1:], strict=False):
        a = _spearman_of(prev)
        b = _spearman_of(nxt)
        if a is None or b is None:
            deltas.append(
                AblationDelta(
                    from_stage=prev.stage_name,
                    to_stage=nxt.stage_name,
                    spearman_delta=None,
                    improved=None,
                )
            )
        else:
            delta = round(b - a, 6)
            deltas.append(
                AblationDelta(
                    from_stage=prev.stage_name,
                    to_stage=nxt.stage_name,
                    spearman_delta=delta,
                    improved=delta > 0,  # worse (<=0) reported plainly, not softened
                )
            )
    return deltas


def run_ablation_study(
    ground_truth_dataset: GroundTruthDataset,
    stages: list[StageScorer],
    resolver: PairResolver,
) -> AblationReport:
    """Run stages 1-4 (5 marked not_available) over the ground truth and report.

    REFUSES with status='cannot_run' if the ground truth has no reconciled pairs,
    rather than running against synthetic data.
    """
    reconciled = _reconciled_pairs(ground_truth_dataset)
    if not reconciled:
        return AblationReport(
            status="cannot_run",
            message=CANNOT_RUN_MESSAGE,
            ground_truth_n=0,
        )

    stage_results = [_run_stage(s, ground_truth_dataset, resolver) for s in stages]
    return AblationReport(
        status="completed",
        message="Ablation completed. Deltas report actual per-stage change (a "
        "negative delta means that stage did NOT improve accuracy — reported as-is).",
        ground_truth_n=len(reconciled),
        stages=stage_results,
        deltas=_compute_deltas(stage_results),
    )


def build_default_stages(
    tfidf_scorer: TFIDFScorer,
    embedding_scorer: EmbeddingScorer,
    hybrid_scorer: HybridScorer,
    skill_matcher: SkillMatcher,
    taxonomy_entries: list[SkillTaxonomyEntry],
) -> list[StageScorer]:
    """Wire the five real stages in order (stage 5 = honest placeholder)."""
    return [
        TfidfOnlyStage(tfidf_scorer),
        EmbeddingOnlyStage(embedding_scorer),
        HybridStage(hybrid_scorer),
        HybridRagStage(hybrid_scorer, skill_matcher, taxonomy_entries),
        FullMlStage(),
    ]

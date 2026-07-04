"""Tests for the Phase 5.3 ablation study runner.

All rater/prediction values here are SYNTHETIC unit-test fixtures verifying the
runner's MECHANICS — never real study data. The real study cannot run until
Phase 5.1 has real ratings (the readiness-refusal test proves the runner enforces
exactly that).
"""

from __future__ import annotations

from app.schemas.parsing import ParsedJobDescription, ParsedResume
from app.services.evaluation.ablation_study import (
    FullMlStage,
    run_ablation_study,
)
from app.services.evaluation.ground_truth_schema import (
    GroundTruthDataset,
    GroundTruthPair,
)


def _gt_pair(
    pair_id: str, reconciled: float | None, reconciled_flag=True
) -> GroundTruthPair:
    return GroundTruthPair(
        pair_id=pair_id,
        resume_id=f"r-{pair_id}",
        jd_id=f"j-{pair_id}",
        case_type="ambiguous",
        reconciled_score=reconciled,
        status="reconciled" if reconciled_flag else "awaiting_raters",
    )


def _reconciled_dataset() -> GroundTruthDataset:
    return GroundTruthDataset(
        pairs=[
            _gt_pair("p1", 90.0),
            _gt_pair("p2", 70.0),
            _gt_pair("p3", 50.0),
            _gt_pair("p4", 30.0),
        ]
    )


class _StubResolver:
    """Returns parsed objects whose raw_text encodes the resume_id (for lookup)."""

    def __init__(self, fail_for: set[str] | None = None) -> None:
        self._fail_for = fail_for or set()

    def resolve(
        self, resume_id: str, jd_id: str
    ) -> tuple[ParsedResume, ParsedJobDescription]:
        if resume_id in self._fail_for:
            raise ValueError(f"resolver could not parse {resume_id}")
        resume = ParsedResume(
            raw_text=resume_id,
            contact_info_present=False,
            parsing_confidence=1.0,
            pipeline_version="parser-v1",
        )
        jd = ParsedJobDescription(
            raw_text=jd_id, parsing_confidence=1.0, pipeline_version="parser-v1"
        )
        return resume, jd


class _StubStage:
    """Deterministic stage: returns a preset score per resume raw_text."""

    def __init__(self, name: str, version: str, scores: dict[str, float]) -> None:
        self.name = name
        self.pipeline_version = version
        self.available = True
        self._scores = scores

    def score(self, resume: ParsedResume, jd: ParsedJobDescription) -> float:
        return self._scores[resume.raw_text]


# Perfectly-ordered stage (Spearman 1.0 vs GT order p1>p2>p3>p4).
_PERFECT = {"r-p1": 90.0, "r-p2": 70.0, "r-p3": 50.0, "r-p4": 30.0}
# Reversed stage (Spearman -1.0) — a genuinely WORSE stage.
_REVERSED = {"r-p1": 30.0, "r-p2": 50.0, "r-p3": 70.0, "r-p4": 90.0}


# --- Readiness refusal (highest priority) ------------------------------------


def test_refuses_on_unready_ground_truth() -> None:
    awaiting = GroundTruthDataset(
        pairs=[_gt_pair("p1", None, reconciled_flag=False)],
        notes="AWAITING REAL RATER INPUT",
    )
    report = run_ablation_study(
        awaiting, [_StubStage("s1", "v1-tfidf", _PERFECT)], _StubResolver()
    )
    assert report.status == "cannot_run"
    assert "GROUND TRUTH NOT YET COLLECTED" in report.message
    assert report.stages == []


def test_refuses_on_empty_dataset() -> None:
    report = run_ablation_study(GroundTruthDataset(), [], _StubResolver())
    assert report.status == "cannot_run"


# --- Per-stage invocation + report -------------------------------------------


def test_runs_each_stage_and_bundles_reports() -> None:
    stages = [
        _StubStage("s1", "v1-tfidf", _PERFECT),
        _StubStage("s2", "v2-embeddings", _REVERSED),
    ]
    report = run_ablation_study(_reconciled_dataset(), stages, _StubResolver())
    assert report.status == "completed"
    assert report.ground_truth_n == 4
    assert len(report.stages) == 2
    assert all(s.status == "completed" and s.report is not None for s in report.stages)
    # Each stage's full EvaluationReport survives (caveat present).
    assert "PROOF-OF-CONCEPT SCALE" in report.stages[0].report.small_sample_caveat


def test_delta_reports_a_worse_stage_plainly() -> None:
    stages = [
        _StubStage("s1", "v1-tfidf", _PERFECT),  # Spearman 1.0
        _StubStage("s2", "v2-embeddings", _REVERSED),  # Spearman -1.0
    ]
    report = run_ablation_study(_reconciled_dataset(), stages, _StubResolver())
    delta = report.deltas[0]
    assert delta.spearman_delta == -2.0  # 1.0 -> -1.0, reported as-is
    assert delta.improved is False  # worse is not softened


def test_delta_reports_improvement() -> None:
    stages = [
        _StubStage("s1", "v1-tfidf", _REVERSED),  # -1.0
        _StubStage("s2", "v2-embeddings", _PERFECT),  # 1.0
    ]
    report = run_ablation_study(_reconciled_dataset(), stages, _StubResolver())
    assert report.deltas[0].spearman_delta == 2.0
    assert report.deltas[0].improved is True


# --- Stage 5 honesty ---------------------------------------------------------


def test_stage5_is_not_available_never_fabricated() -> None:
    stages = [_StubStage("s1", "v1-tfidf", _PERFECT), FullMlStage()]
    report = run_ablation_study(_reconciled_dataset(), stages, _StubResolver())
    stage5 = report.stages[-1]
    assert stage5.status == "not_available"
    assert stage5.report is None  # no substituted numbers
    # Not included in deltas (only completed stages).
    assert all(d.to_stage != stage5.stage_name for d in report.deltas)


# --- Per-pair failure recording ----------------------------------------------


def test_per_pair_failure_recorded_not_silently_skipped() -> None:
    stages = [_StubStage("s1", "v1-tfidf", _PERFECT)]
    resolver = _StubResolver(fail_for={"r-p3"})  # one pair fails to resolve
    report = run_ablation_study(_reconciled_dataset(), stages, resolver)
    stage = report.stages[0]
    assert len(stage.failures) == 1
    assert stage.failures[0].pair_id == "p3"
    assert "could not parse" in stage.failures[0].error
    # Effective sample size dropped from 4 to 3 and is visible.
    assert stage.report is not None and stage.report.n == 3
    assert "effective n=3" in stage.note


# --- Determinism -------------------------------------------------------------


def test_ablation_is_deterministic() -> None:
    stages = [
        _StubStage("s1", "v1-tfidf", _PERFECT),
        _StubStage("s2", "v2-embeddings", _REVERSED),
    ]
    a = run_ablation_study(_reconciled_dataset(), stages, _StubResolver())
    b = run_ablation_study(_reconciled_dataset(), stages, _StubResolver())
    assert a.model_dump() == b.model_dump()

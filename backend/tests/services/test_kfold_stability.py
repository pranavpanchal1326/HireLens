"""Tests for the Phase 5.4 k-fold + seed-variance stability module.

All scores here are SYNTHETIC unit-test fixtures verifying MECHANICS — never real
study data. The refusal test proves real runs are blocked until 5.1 has real data.
"""

from __future__ import annotations

from app.services.evaluation.ground_truth_schema import (
    GroundTruthDataset,
    GroundTruthPair,
)
from app.services.evaluation.kfold_stability import (
    _aggregate,
    generate_kfolds,
    run_kfold_validation,
    run_seed_variance_test,
)


def _pair(pid: str, score: float, case: str, ready=True) -> GroundTruthPair:
    return GroundTruthPair(
        pair_id=pid,
        resume_id=f"r-{pid}",
        jd_id=f"j-{pid}",
        case_type=case,  # type: ignore[arg-type]
        reconciled_score=score,
        status="reconciled" if ready else "awaiting_raters",
    )


def _balanced_dataset() -> GroundTruthDataset:
    # 9 pairs, 3 per case_type — allows genuine stratification at k=3.
    pairs = []
    for i, case in enumerate(["clear_fit", "clear_gap", "ambiguous"]):
        for j in range(3):
            pairs.append(_pair(f"{case}{j}", 90 - (i * 30) - j, case))
    return GroundTruthDataset(pairs=pairs)


class _StubFoldScorer:
    """Deterministic: returns a preset score per pair_id (fit is a no-op)."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.fit_calls = 0

    def fit(self, train_pairs: list[GroundTruthPair]) -> None:
        self.fit_calls += 1

    def predict(self, test_pairs: list[GroundTruthPair]):
        from app.services.evaluation.kfold_stability import _prediction

        return [
            _prediction(p.resume_id, p.jd_id, self._scores[p.pair_id])
            for p in test_pairs
        ]


# --- _aggregate (hand-computed) ----------------------------------------------


def test_aggregate_hand_values() -> None:
    out = _aggregate([1.0, 0.0, -1.0])
    assert out["mean"] == 0.0
    assert out["std"] == 1.0  # sample stdev of {1,0,-1} = sqrt(2/2) = 1.0
    assert out["n"] == 3


def test_aggregate_single_value_std_is_none() -> None:
    out = _aggregate([0.8])
    assert out["mean"] == 0.8 and out["std"] is None  # never faked as 0


def test_aggregate_two_equal_values_std_zero() -> None:
    out = _aggregate([0.5, 0.5])
    assert out["mean"] == 0.5 and out["std"] == 0.0


# --- fold generation + achieved stratification -------------------------------


def test_stratified_folds_when_classes_allow() -> None:
    folds = generate_kfolds(_balanced_dataset(), k=3, seed=42)
    assert len(folds) == 3
    assert all(f.stratified for f in folds)
    # Each test fold should carry all 3 case types (1 each) under stratification.
    for f in folds:
        assert set(f.test_case_type_counts) == {"clear_fit", "clear_gap", "ambiguous"}


def test_fallback_to_unstratified_reported_honestly() -> None:
    # A class smaller than k → stratification impossible → reported stratified=False.
    pairs = (
        [_pair(f"cf{i}", 80 - i, "clear_fit") for i in range(4)]
        + [_pair("cg0", 20, "clear_gap")]
        + [_pair("am0", 50, "ambiguous")]
    )
    folds = generate_kfolds(GroundTruthDataset(pairs=pairs), k=3, seed=1)
    assert all(f.stratified is False for f in folds)  # not claimed as stratified


# --- k-fold runner -----------------------------------------------------------


def _perfect_scores(ds: GroundTruthDataset) -> dict[str, float]:
    # Predicted == reconciled → strong per-fold correlation.
    return {p.pair_id: float(p.reconciled_score) for p in ds.pairs}  # type: ignore[arg-type]


def test_kfold_reports_mean_and_std_never_mean_only() -> None:
    ds = _balanced_dataset()
    report = run_kfold_validation(
        _StubFoldScorer(_perfect_scores(ds)), ds, k=3, seed=42
    )
    assert report.status == "completed"
    assert report.mean_spearman is not None
    assert report.std_spearman is not None  # std NEVER dropped when >=2 folds
    assert report.n_folds_with_spearman >= 2
    assert "test folds contained" in report.fold_size_note
    assert "PROOF-OF-CONCEPT SCALE" in report.small_sample_caveat


def test_kfold_fit_called_once_per_fold() -> None:
    ds = _balanced_dataset()
    scorer = _StubFoldScorer(_perfect_scores(ds))
    run_kfold_validation(scorer, ds, k=3, seed=42)
    assert scorer.fit_calls == 3  # fit on each fold's train split


def test_kfold_is_deterministic() -> None:
    ds = _balanced_dataset()
    a = run_kfold_validation(_StubFoldScorer(_perfect_scores(ds)), ds, k=3, seed=42)
    b = run_kfold_validation(_StubFoldScorer(_perfect_scores(ds)), ds, k=3, seed=42)
    assert a.model_dump() == b.model_dump()


# --- seed-variance runner (two-layer) ----------------------------------------


def test_seed_variance_reports_two_layers() -> None:
    ds = _balanced_dataset()
    report = run_seed_variance_test(
        _StubFoldScorer(_perfect_scores(ds)), ds, k=3, seed_list=(1, 2, 3)
    )
    assert report.status == "completed"
    assert len(report.per_seed) == 3
    # fold-level spread within each seed present...
    assert all(s.std_spearman_across_folds is not None for s in report.per_seed)
    # ...and seed-level spread across seeds present (never mean-only).
    assert report.across_seed_mean_spearman is not None
    assert report.across_seed_std_spearman is not None


def test_seed_variance_is_deterministic() -> None:
    ds = _balanced_dataset()
    a = run_seed_variance_test(
        _StubFoldScorer(_perfect_scores(ds)), ds, seed_list=(1, 2)
    )
    b = run_seed_variance_test(
        _StubFoldScorer(_perfect_scores(ds)), ds, seed_list=(1, 2)
    )
    assert a.model_dump() == b.model_dump()


# --- readiness refusal (primary honesty gate) --------------------------------


def test_kfold_refuses_on_unready_ground_truth() -> None:
    awaiting = GroundTruthDataset(pairs=[_pair("p1", 50, "ambiguous", ready=False)])
    report = run_kfold_validation(_StubFoldScorer({}), awaiting, k=3, seed=42)
    assert report.status == "cannot_run"
    assert "NOT YET COLLECTED" in report.message
    assert report.mean_spearman is None and report.std_spearman is None


def test_seed_variance_refuses_on_unready_ground_truth() -> None:
    report = run_seed_variance_test(_StubFoldScorer({}), GroundTruthDataset())
    assert report.status == "cannot_run"
    assert report.across_seed_std_spearman is None


def test_kfold_refuses_gracefully_on_single_reconciled_pair() -> None:
    # 1 reconciled pair passes readiness (>0) but cannot be folded — must refuse
    # cleanly, not crash inside generate_kfolds.
    one = GroundTruthDataset(pairs=[_pair("only", 50, "ambiguous")])
    report = run_kfold_validation(_StubFoldScorer({"only": 50.0}), one, k=3, seed=1)
    assert report.status == "cannot_run"
    assert "INSUFFICIENT DATA" in report.message
    assert report.n == 1

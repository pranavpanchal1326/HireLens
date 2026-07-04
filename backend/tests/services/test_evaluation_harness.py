"""Tests for the Phase 5.2 evaluation harness.

Each metric is checked against an INDEPENDENTLY hand-computed expected value, not
just "runs without error" — this is the most-graded artifact, so a plausible wrong
number must not slip through.
"""

from __future__ import annotations

from app.schemas.scoring import ConfidenceLevel, FeatureVector, ScoreResult
from app.services.evaluation.evaluation_harness import (
    compute_classification_metrics,
    compute_ndcg,
    compute_precision_at_k,
    compute_spearman,
    evaluate,
)
from app.services.evaluation.ground_truth_schema import (
    GroundTruthDataset,
    GroundTruthPair,
)


def _pred(resume_id: str, jd_id: str, score: int) -> ScoreResult:
    return ScoreResult(
        resume_id=resume_id,
        jd_id=jd_id,
        final_score=score,
        feature_vector=FeatureVector(
            tfidf_score=0.0,
            embedding_score=0.0,
            skill_overlap_pct=0.0,
            exp_match=0.0,
            edu_match=0.0,
        ),
        scoring_confidence=0.5,
        confidence_level=ConfidenceLevel.MEDIUM,
        parsing_confidence=0.8,
        pipeline_version="v3-hybrid",
    )


def _gt_pair(pair_id: str, reconciled: float, case="ambiguous") -> GroundTruthPair:
    return GroundTruthPair(
        pair_id=pair_id,
        resume_id=f"r-{pair_id}",
        jd_id=f"j-{pair_id}",
        case_type=case,  # type: ignore[arg-type]
        reconciled_score=reconciled,
        status="reconciled",
    )


# --- Spearman (hand-checked) -------------------------------------------------


def test_spearman_perfect_positive() -> None:
    out = compute_spearman([1, 2, 3, 4], [10, 20, 30, 40])
    assert out["correlation"] == 1.0
    assert out["n"] == 4


def test_spearman_perfect_negative() -> None:
    out = compute_spearman([1, 2, 3, 4], [40, 30, 20, 10])
    assert out["correlation"] == -1.0


def test_spearman_n_below_two_is_none() -> None:
    out = compute_spearman([5], [9])
    assert out["correlation"] is None and out["n"] == 1


def test_spearman_all_tied_is_none() -> None:
    # Zero variance on one side → correlation undefined (not a fake 0.0).
    out = compute_spearman([50, 50, 50], [10, 20, 30])
    assert out["correlation"] is None


# --- Precision@k (hand-checked) ----------------------------------------------


def test_precision_at_k_hand_value() -> None:
    # GT top-2 = {a, b}. Predicted top-2 by score = {a, c}. Overlap {a} → 1/2.
    gt = {"a": 90.0, "b": 80.0, "c": 70.0, "d": 10.0}
    pred = {"a": 95.0, "b": 10.0, "c": 85.0, "d": 5.0}
    out = compute_precision_at_k(pred, gt, k=2)
    assert out["precision"] == 0.5
    assert out["k_effective"] == 2


def test_precision_at_k_shrinks_k_to_n() -> None:
    out = compute_precision_at_k({"a": 1.0}, {"a": 1.0}, k=5)
    assert out["k_effective"] == 1 and out["precision"] == 1.0


# --- NDCG (hand-checked) -----------------------------------------------------


def test_ndcg_ideal_ordering_is_one() -> None:
    gt = {"a": 3.0, "b": 2.0, "c": 1.0}
    pred = {"a": 3.0, "b": 2.0, "c": 1.0}  # same order as ideal
    assert compute_ndcg(pred, gt)["ndcg"] == 1.0


def test_ndcg_reverse_ordering_hand_value() -> None:
    # true gains [3,2,1], predicted order reversed → DCG 3.76186 / IDCG 4.76186
    # = 0.78992 (hand-computed with log2 discounting).
    gt = {"a": 3.0, "b": 2.0, "c": 1.0}
    pred = {"a": 1.0, "b": 2.0, "c": 3.0}  # ranks c, b, a
    ndcg = compute_ndcg(pred, gt)["ndcg"]
    assert ndcg is not None and abs(ndcg - 0.78992) < 0.001


# --- Classification (explicit threshold, hand-checked) -----------------------


def test_classification_perfect_at_threshold() -> None:
    out = compute_classification_metrics(
        [80, 40, 90, 20], [85, 30, 88, 10], threshold=50
    )
    assert out["accuracy"] == 1.0 and out["f1"] == 1.0
    assert out["confusion_matrix"] == [[2, 0], [0, 2]]  # [[TN,FP],[FN,TP]]
    assert out["threshold"] == 50


def test_classification_threshold_changes_labels() -> None:
    # predicted says fit/nofit; ground truth is opposite at threshold 50 → acc 0.
    out = compute_classification_metrics([80, 40], [40, 80], threshold=50)
    assert out["accuracy"] == 0.0


# --- evaluate() unified report -----------------------------------------------


def _dataset() -> GroundTruthDataset:
    pairs = [
        _gt_pair("p1", 90.0, "clear_fit"),
        _gt_pair("p2", 85.0, "clear_fit"),
        _gt_pair("p3", 20.0, "clear_gap"),
        _gt_pair("p4", 15.0, "clear_gap"),
        _gt_pair("p5", 55.0, "ambiguous"),
        _gt_pair("p6", 45.0, "ambiguous"),
    ]
    return GroundTruthDataset(pairs=pairs)


def _predictions() -> list[ScoreResult]:
    ds = _dataset()
    # Predicted scores loosely track GT so metrics are meaningful.
    order = {"p1": 88, "p2": 80, "p3": 25, "p4": 10, "p5": 60, "p6": 40}
    return [_pred(p.resume_id, p.jd_id, order[p.pair_id]) for p in ds.pairs]


def test_evaluate_always_carries_small_sample_caveat() -> None:
    report = evaluate(_predictions(), _dataset())
    assert report.n == 6
    assert "PROOF-OF-CONCEPT SCALE" in report.small_sample_caveat
    assert "n=6" in report.small_sample_caveat


def test_evaluate_per_case_type_breakdown_present() -> None:
    report = evaluate(_predictions(), _dataset())
    assert set(report.per_case_type) == {"clear_fit", "clear_gap", "ambiguous"}
    assert report.per_case_type["ambiguous"]["n"] == 2
    assert "spearman" in report.per_case_type["ambiguous"]
    assert "precision_at_5" in report.per_case_type["ambiguous"]


def test_evaluate_classification_only_when_threshold_given() -> None:
    assert evaluate(_predictions(), _dataset()).classification is None
    with_thr = evaluate(_predictions(), _dataset(), classification_threshold=50)
    assert with_thr.classification is not None
    assert with_thr.classification["threshold"] == 50


def test_evaluate_is_deterministic() -> None:
    a = evaluate(_predictions(), _dataset(), classification_threshold=50)
    b = evaluate(_predictions(), _dataset(), classification_threshold=50)
    assert a.model_dump() == b.model_dump()


def test_evaluate_counts_unmatched_predictions() -> None:
    preds = _predictions() + [_pred("ghost", "ghost", 50)]  # no GT pair
    report = evaluate(preds, _dataset())
    assert report.n_predictions_unmatched == 1
    assert report.n == 6  # unmatched excluded from metrics

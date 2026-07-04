"""Evaluation harness (Phase 5.2) — Spearman, Precision@k, NDCG, classification.

The project's single most-graded artifact (PRD §7). Computes ranking-quality
metrics comparing the system's predicted scores against Phase 5.1's reconciled
human ground truth. Correctness + determinism matter more here than anywhere: a
subtly-wrong metric returns a plausible-but-false number every later phase
inherits.

Design decisions:
  - Standard statistical implementations are used (scipy for Spearman, scikit-learn
    for NDCG + classification) rather than hand-rolled math — reinventing these on
    the most-graded artifact is needless risk. Precision@k is small/explicit enough
    to implement directly, with a stated relevance definition + tie-breaking.
  - EVERY report carries the sample size and a MANDATORY small-sample caveat
    (PRD §7.3 / Design Blueprint P3). There is no "clean" mode that strips it.
  - Tie-breaking is explicit: ranking selections sort by (-score, id) so equal
    scores never depend on incidental sort stability (determinism, PRD §7).

No LLM anywhere. Consumes Phase 0.2 ScoreResult + Phase 5.1 GroundTruthDataset
as-is; does not modify either.
"""

from __future__ import annotations

import warnings

import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    ndcg_score,
    precision_score,
    recall_score,
)

from app.schemas.scoring import ScoreResult
from app.services.evaluation.ground_truth_schema import GroundTruthDataset

DEFAULT_PRECISION_KS = (5, 10)
_ROUND = 6


def small_sample_caveat(n: int) -> str:
    """The mandatory caveat attached to every report (never omittable)."""
    return (
        f"PROOF-OF-CONCEPT SCALE: n={n} reconciled pair(s). Metrics are indicative "
        f"only — interpret with k-fold cross-validation (Phase 5.4) and wide "
        f"confidence intervals; NOT production-validated (PRD §7.3)."
    )


def _rank_ids_by_score(scores: dict[str, float]) -> list[str]:
    """Ids ordered best-first. Explicit tie-break: higher score first, then id
    ascending — deterministic regardless of input/sort stability."""
    return sorted(scores, key=lambda i: (-scores[i], i))


def compute_spearman(
    predicted_scores: list[float], ground_truth_scores: list[float]
) -> dict[str, float | int | None]:
    """Spearman rank correlation (scipy) between predicted and GT scores.

    Returns {correlation, p_value, n}. correlation/p_value are None when n<2 or a
    correlation is undefined (e.g. all-tied scores → zero variance).
    """
    n = len(predicted_scores)
    if n != len(ground_truth_scores):
        raise ValueError("predicted and ground-truth lengths differ.")
    if n < 2:
        return {"correlation": None, "p_value": None, "n": n}
    with warnings.catch_warnings():
        # All-tied input is a case we handle explicitly below (→ None); scipy's
        # ConstantInputWarning here is expected noise, not an error.
        warnings.simplefilter("ignore", ConstantInputWarning)
        result = spearmanr(predicted_scores, ground_truth_scores)
    corr = float(result.correlation)
    pval = float(result.pvalue)
    if np.isnan(corr):  # zero variance on one side → correlation undefined
        return {"correlation": None, "p_value": None, "n": n}
    # p-value is nan for n==2 (undefined); never emit nan — it breaks determinism
    # and JSON. Report None instead so the correlation still stands on its own.
    p_out = None if np.isnan(pval) else round(pval, _ROUND)
    return {"correlation": round(corr, _ROUND), "p_value": p_out, "n": n}


def compute_precision_at_k(
    predicted_ranking: dict[str, float],
    ground_truth_ranking: dict[str, float],
    k: int,
) -> dict[str, float | int | None]:
    """Precision@k over items shared by both rankings.

    RELEVANCE DEFINITION (explicit, fixed): the "relevant" set is the top-k items
    by GROUND-TRUTH score; precision@k = (how many of the system's top-k are in
    that GT top-k) / k_effective, where k_effective = min(k, n_common). Ties are
    broken by (-score, id). Returns {precision, k, k_effective, n_total}.
    """
    common = sorted(set(predicted_ranking) & set(ground_truth_ranking))
    n_total = len(common)
    if n_total == 0:
        return {"precision": None, "k": k, "k_effective": 0, "n_total": 0}
    k_eff = min(k, n_total)
    pred = {i: predicted_ranking[i] for i in common}
    gt = {i: ground_truth_ranking[i] for i in common}
    relevant = set(_rank_ids_by_score(gt)[:k_eff])
    retrieved = set(_rank_ids_by_score(pred)[:k_eff])
    precision = len(relevant & retrieved) / k_eff
    return {
        "precision": round(precision, _ROUND),
        "k": k,
        "k_effective": k_eff,
        "n_total": n_total,
    }


def compute_ndcg(
    predicted_ranking: dict[str, float],
    ground_truth_ranking: dict[str, float],
    k: int | None = None,
) -> dict[str, float | int | None]:
    """NDCG (scikit-learn) of the system's ordering, graded by GT relevance.

    Standard formula: DCG = sum_i gain_i / log2(i+1) with LINEAR gains (the GT
    scores themselves), normalized by the ideal DCG. GT scores are the relevance
    gains; the system's scores induce the ranking. Returns {ndcg, k, n_total}.
    None when n<2 (NDCG undefined for a single item).
    """
    common = sorted(set(predicted_ranking) & set(ground_truth_ranking))
    n_total = len(common)
    if n_total < 2:
        return {"ndcg": None, "k": k, "n_total": n_total}
    y_true = np.array([[ground_truth_ranking[i] for i in common]], dtype=float)
    y_score = np.array([[predicted_ranking[i] for i in common]], dtype=float)
    value = float(ndcg_score(y_true, y_score, k=k))
    return {"ndcg": round(value, _ROUND), "k": k, "n_total": n_total}


def compute_classification_metrics(
    predicted_scores: list[float],
    ground_truth_scores: list[float],
    threshold: float,
) -> dict[str, object]:
    """Binarize both score lists at ``threshold`` (score >= threshold → good-fit=1)
    and compute accuracy/precision/recall/F1/confusion matrix (scikit-learn).

    ``threshold`` is a REQUIRED, explicit parameter — where the fit/no-fit line is
    drawn changes every number, so it is never a hidden default. confusion_matrix
    is returned as [[TN, FP], [FN, TP]]. Returns {accuracy, precision, recall, f1,
    confusion_matrix, threshold, n}.
    """
    n = len(predicted_scores)
    if n != len(ground_truth_scores):
        raise ValueError("predicted and ground-truth lengths differ.")
    y_pred = [1 if s >= threshold else 0 for s in predicted_scores]
    y_true = [1 if s >= threshold else 0 for s in ground_truth_scores]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), _ROUND),
        "precision": round(
            float(precision_score(y_true, y_pred, zero_division=0)), _ROUND
        ),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), _ROUND),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), _ROUND),
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        "threshold": threshold,
        "n": n,
    }


class EvaluationReport(BaseModel):
    """Bundled metrics + mandatory context. Every field of context is required."""

    n: int
    small_sample_caveat: str
    spearman: dict[str, float | int | None]
    precision_at_5: dict[str, float | int | None]
    precision_at_10: dict[str, float | int | None]
    ndcg: dict[str, float | int | None]
    classification: dict[str, object] | None = None
    per_case_type: dict[str, dict[str, object]] = Field(default_factory=dict)
    n_predictions_unmatched: int = 0


def _aligned(
    predictions: list[ScoreResult], ground_truth_dataset: GroundTruthDataset
) -> tuple[list[str], list[float], list[float], list[str], int]:
    """Match predictions to reconciled GT pairs by (resume_id, jd_id).

    Returns (pair_ids, predicted_scores, gt_scores, case_types, n_unmatched), all
    aligned and sorted by pair_id for deterministic output. Only pairs with a
    non-None reconciled_score are included.
    """
    gt_by_key = {
        (p.resume_id, p.jd_id): p
        for p in ground_truth_dataset.pairs
        if p.reconciled_score is not None
    }
    matched: list[tuple[str, float, float, str]] = []
    unmatched = 0
    for pred in predictions:
        pair = gt_by_key.get((pred.resume_id, pred.jd_id))
        if pair is None or pair.reconciled_score is None:
            unmatched += 1
            continue
        matched.append(
            (
                pair.pair_id,
                float(pred.final_score),
                pair.reconciled_score,
                pair.case_type,
            )
        )
    matched.sort(key=lambda t: t[0])  # deterministic ordering by pair_id
    ids = [m[0] for m in matched]
    preds = [m[1] for m in matched]
    gts = [m[2] for m in matched]
    cases = [m[3] for m in matched]
    return ids, preds, gts, cases, unmatched


def evaluate(
    predictions: list[ScoreResult],
    ground_truth_dataset: GroundTruthDataset,
    classification_threshold: float | None = None,
) -> EvaluationReport:
    """Unified entry point → EvaluationReport (always carries the caveat + n).

    Matches each prediction to a reconciled GT pair by (resume_id, jd_id), then
    computes Spearman, Precision@5/@10, NDCG, an optional classification block
    (only when ``classification_threshold`` is explicitly provided), and a
    per-case_type breakdown of Spearman + Precision@5 (the payoff of Phase 5.1's
    ambiguous-case quota).
    """
    ids, preds, gts, cases, unmatched = _aligned(predictions, ground_truth_dataset)
    n = len(ids)
    pred_map = dict(zip(ids, preds, strict=True))
    gt_map = dict(zip(ids, gts, strict=True))

    classification = (
        compute_classification_metrics(preds, gts, classification_threshold)
        if classification_threshold is not None
        else None
    )

    per_case_type: dict[str, dict[str, object]] = {}
    for case in ("clear_fit", "clear_gap", "ambiguous"):
        idx = [i for i, c in enumerate(cases) if c == case]
        if not idx:
            continue
        c_ids = [ids[i] for i in idx]
        c_pred = {i: pred_map[i] for i in c_ids}
        c_gt = {i: gt_map[i] for i in c_ids}
        per_case_type[case] = {
            "n": len(idx),
            "spearman": compute_spearman(
                [preds[i] for i in idx], [gts[i] for i in idx]
            ),
            "precision_at_5": compute_precision_at_k(c_pred, c_gt, 5),
        }

    return EvaluationReport(
        n=n,
        small_sample_caveat=small_sample_caveat(n),
        spearman=compute_spearman(preds, gts),
        precision_at_5=compute_precision_at_k(pred_map, gt_map, 5),
        precision_at_10=compute_precision_at_k(pred_map, gt_map, 10),
        ndcg=compute_ndcg(pred_map, gt_map),
        classification=classification,
        per_case_type=per_case_type,
        n_predictions_unmatched=unmatched,
    )

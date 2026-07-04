"""Multi-rater reconciliation (Phase 5.1).

Implements PRD §6's mandated process: given N independent raters' blind scores for
the same pairs, compute reconciled ground-truth scores AND an explicit inter-rater
reliability metric — never a naive mean alone. Sharp disagreements are FLAGGED and
preserved, not silently averaged away (Design Blueprint P3).

No LLM is involved in producing or influencing any score.
"""

from __future__ import annotations

from app.services.evaluation.ground_truth_schema import (
    GroundTruthDataset,
    GroundTruthPair,
)

# Per-pair range (max-min, on the 0-100 scale) beyond which raters are considered
# to disagree sharply and the pair is flagged for review.
# PROVISIONAL — a defensible starting point; revisit once real rater spread is seen.
DIVERGENCE_THRESHOLD = 20.0
_SCORE_PRECISION = 2
_AGREEMENT_PRECISION = 4


def _pearson(x: list[float], y: list[float]) -> float | None:
    """Pearson correlation of two equal-length score lists. None if undefined
    (fewer than 2 points, or either list has zero variance)."""
    n = len(x)
    if n != len(y) or n < 2:
        return None
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    dx = [xi - mean_x for xi in x]
    dy = [yi - mean_y for yi in y]
    num = sum(a * b for a, b in zip(dx, dy, strict=True))
    den = (sum(a * a for a in dx) * sum(b * b for b in dy)) ** 0.5
    if den == 0.0:
        return None  # a rater gave identical scores to everything → undefined
    return float(num / den)


def reconcile_pair(
    pair: GroundTruthPair, divergence_threshold: float = DIVERGENCE_THRESHOLD
) -> GroundTruthPair:
    """Return a copy of ``pair`` with reconciled_score, inter_rater_range, and
    divergence_flag filled from its rater_scores. Reconciliation = naive mean, but
    the range and flag keep any disagreement explicit rather than hidden."""
    scores = [rs.score for rs in pair.rater_scores]
    if not scores:
        return pair.model_copy(
            update={"reconciled_score": None, "status": "awaiting_raters"}
        )
    reconciled = round(sum(scores) / len(scores), _SCORE_PRECISION)
    rng = round(max(scores) - min(scores), _SCORE_PRECISION)
    return pair.model_copy(
        update={
            "reconciled_score": reconciled,
            "inter_rater_range": rng,
            "divergence_flag": rng > divergence_threshold,
            "status": "reconciled",
        }
    )


def mean_pairwise_pearson(rater_to_scores: dict[str, list[float]]) -> float | None:
    """Mean Pearson correlation across all rater pairs (overall reliability).

    Each value is one rater's scores aligned across the SAME pairs in the same
    order. Returns None if fewer than 2 raters or no pair yields a defined
    correlation.
    """
    rater_ids = sorted(rater_to_scores)
    if len(rater_ids) < 2:
        return None
    correlations: list[float] = []
    for i in range(len(rater_ids)):
        for j in range(i + 1, len(rater_ids)):
            corr = _pearson(
                rater_to_scores[rater_ids[i]], rater_to_scores[rater_ids[j]]
            )
            if corr is not None:
                correlations.append(corr)
    if not correlations:
        return None
    return round(sum(correlations) / len(correlations), _AGREEMENT_PRECISION)


def reconcile_dataset(
    dataset: GroundTruthDataset, divergence_threshold: float = DIVERGENCE_THRESHOLD
) -> GroundTruthDataset:
    """Reconcile every rated pair and compute overall inter-rater agreement.

    Only pairs where EVERY rater in the dataset scored the pair contribute to the
    overall Pearson metric (so a partially-rated pair can't distort reliability).
    Unrated pairs are left awaiting_raters (never assigned a fabricated score).
    """
    reconciled_pairs = [reconcile_pair(p, divergence_threshold) for p in dataset.pairs]

    # Build aligned per-rater score vectors over fully-rated pairs.
    all_raters = sorted({rs.rater_id for p in dataset.pairs for rs in p.rater_scores})
    rater_to_scores: dict[str, list[float]] = {r: [] for r in all_raters}
    for pair in dataset.pairs:
        by_rater = {rs.rater_id: rs.score for rs in pair.rater_scores}
        if all_raters and all(r in by_rater for r in all_raters):
            for r in all_raters:
                rater_to_scores[r].append(by_rater[r])

    agreement = mean_pairwise_pearson(rater_to_scores) if all_raters else None

    return dataset.model_copy(
        update={
            "pairs": reconciled_pairs,
            "n_raters": len(all_raters) or None,
            "overall_inter_rater_agreement": agreement,
        }
    )

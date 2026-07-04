"""k-Fold cross-validation + seed-variance stability testing (Phase 5.4).

Small-data validation rigor on top of Phase 5.2's harness (PRD §7.3). At n=20-30 a
single split's metric is a lucky/unlucky artifact of which few pairs landed in the
test set; k-fold + seed variance turn a meaningless point estimate into a
defensible "X ± Y, stable across Z seeds" claim.

Non-negotiable honesty carried forward from Phase 5.3:
  - REFUSE to run if the ground truth isn't collected yet (AWAITING state).
  - NEVER report a mean without its standard deviation — the std IS the finding.
  - Report the ACTUAL achieved case_type stratification per fold, not the intent.

Forward-compatibility (do NOT implement Phase 6 here): the runner validates any
object satisfying the FoldScorer protocol. For Phase 5.3's Stages 1-4 the per-fold
fit() is a documented NO-OP (see StageFoldAdapter) because those stages have no
trainable parameter fit on the ground-truth pairs — so here k-fold measures
metric-stability across held-out subsets. Phase 6.2's trained model WILL have a
real per-fold fit; it plugs into this same protocol with no structural rework.

No LLM anywhere. Calls Phase 5.2 evaluate() + reuses Phase 5.3's readiness check.
"""

from __future__ import annotations

import statistics
from typing import Protocol

from pydantic import BaseModel, Field
from sklearn.model_selection import KFold, StratifiedKFold

from app.schemas.scoring import ScoreResult
from app.services.evaluation.ablation_study import PairResolver, is_ground_truth_ready
from app.services.evaluation.evaluation_harness import evaluate, small_sample_caveat
from app.services.evaluation.ground_truth_schema import (
    GroundTruthDataset,
    GroundTruthPair,
)

# k=5 chosen for this project's n≈20-30: 5-fold gives test folds of ~4-6 pairs —
# small but workable. 10-fold would give folds of ~2-3 (too small for a stable
# per-fold Spearman); leave-one-out would give per-fold n=1 (Spearman undefined).
# Named + justified, not a large-dataset default. PROVISIONAL if n moves materially.
DEFAULT_K = 5

# Fixed seed list, chosen BEFORE seeing any results (never cherry-picked post-hoc).
# 5 seeds is enough to expose split-luck at this scale without excessive runtime.
DEFAULT_SEEDS = (11, 23, 42, 71, 101)

_ROUND = 6


class FoldScorer(Protocol):
    """A validatable scoring function. ``fit`` may be a no-op (see module note)."""

    def fit(self, train_pairs: list[GroundTruthPair]) -> None: ...

    def predict(self, test_pairs: list[GroundTruthPair]) -> list[ScoreResult]: ...


class StageFoldAdapter:
    """Adapts a Phase 5.3 StageScorer + resolver into a FoldScorer.

    fit() is a documented NO-OP: Stages 1-4 have no parameter fit on the
    ground-truth pairs — TF-IDF is fit once on the large external corpus (Phase
    2.1), embeddings are pretrained (2.2), hybrid weights are fixed/provisional
    (2.4), and the RAG matcher has no fit (3.3). So k-fold here measures how much
    each metric wobbles across held-out subsets, not train/test generalization of a
    fitted model. Phase 6.2's model overrides fit() with a real per-fold training.
    """

    def __init__(self, stage: object, resolver: PairResolver) -> None:
        self._stage = stage
        self._resolver = resolver

    def fit(self, train_pairs: list[GroundTruthPair]) -> None:
        return None  # NO-OP for Stages 1-4 — see class docstring.

    def predict(self, test_pairs: list[GroundTruthPair]) -> list[ScoreResult]:
        preds: list[ScoreResult] = []
        for pair in test_pairs:
            resume, jd = self._resolver.resolve(pair.resume_id, pair.jd_id)
            value = self._stage.score(resume, jd)  # type: ignore[attr-defined]
            preds.append(_prediction(pair.resume_id, pair.jd_id, value))
        return preds


def _prediction(resume_id: str, jd_id: str, score: float) -> ScoreResult:
    from app.schemas.scoring import ConfidenceLevel, FeatureVector

    return ScoreResult(
        resume_id=resume_id,
        jd_id=jd_id,
        final_score=min(100, max(0, round(score))),
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
        pipeline_version="kfold",
    )


# --- Fold generation ---------------------------------------------------------


class FoldSplit(BaseModel):
    fold_index: int
    train_pair_ids: list[str]
    test_pair_ids: list[str]
    test_case_type_counts: dict[str, int]  # ACHIEVED distribution, honestly reported
    stratified: bool


def generate_kfolds(
    dataset: GroundTruthDataset, k: int = DEFAULT_K, seed: int = 42
) -> list[FoldSplit]:
    """Generate k stratified-by-case_type folds over the reconciled pairs.

    Stratifies by case_type when every class has >= k members; otherwise falls
    back to plain KFold and marks those folds stratified=False (never claims
    stratification that didn't happen). k is clamped to n when n < k.
    """
    pairs = [
        p
        for p in dataset.pairs
        if p.status == "reconciled" and p.reconciled_score is not None
    ]
    n = len(pairs)
    if n < 2:
        raise ValueError("Need >= 2 reconciled pairs to build folds.")
    eff_k = min(k, n)
    case_types = [p.case_type for p in pairs]

    # Strict stratification requires every class to have >= eff_k members. We
    # pre-check this explicitly: modern sklearn StratifiedKFold does NOT raise when
    # a class is too small — it only WARNS and does a best-effort split — so relying
    # on an exception would silently mislabel an unstratifiable split as stratified.
    class_counts = {c: case_types.count(c) for c in set(case_types)}
    stratifiable = min(class_counts.values()) >= eff_k

    if stratifiable:
        splitter = StratifiedKFold(n_splits=eff_k, shuffle=True, random_state=seed)
        split_iter = list(splitter.split(range(n), case_types))
        stratified = True
    else:
        # Strict stratification impossible at this size — report reality, not intent.
        kf = KFold(n_splits=eff_k, shuffle=True, random_state=seed)
        split_iter = list(kf.split(range(n)))
        stratified = False

    folds: list[FoldSplit] = []
    for i, (train_idx, test_idx) in enumerate(split_iter):
        counts: dict[str, int] = {}
        for j in test_idx:
            counts[case_types[j]] = counts.get(case_types[j], 0) + 1
        folds.append(
            FoldSplit(
                fold_index=i,
                train_pair_ids=[pairs[j].pair_id for j in train_idx],
                test_pair_ids=[pairs[j].pair_id for j in test_idx],
                test_case_type_counts=counts,
                stratified=stratified,
            )
        )
    return folds


# --- Aggregation -------------------------------------------------------------


def _aggregate(values: list[float]) -> dict[str, float | int | None]:
    """Return {mean, std, n}. std is SAMPLE std (ddof=1); None when n<2 (a spread
    over a single value is undefined and must not be faked as 0)."""
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "n": 0}
    mean = round(sum(values) / n, _ROUND)
    std = round(statistics.stdev(values), _ROUND) if n >= 2 else None
    return {"mean": mean, "std": std, "n": n}


# --- Reports -----------------------------------------------------------------


class FoldMetrics(BaseModel):
    fold_index: int
    n_test: int
    spearman: float | None
    precision_at_5: float | None
    stratified: bool
    test_case_type_counts: dict[str, int]


class KFoldReport(BaseModel):
    status: str  # "completed" | "cannot_run"
    message: str
    k: int
    seed: int
    n: int
    stratified_all_folds: bool
    fold_size_min: int
    fold_size_max: int
    mean_spearman: float | None
    std_spearman: float | None  # NEVER dropped — the headline stability finding
    n_folds_with_spearman: int
    per_fold: list[FoldMetrics] = Field(default_factory=list)
    small_sample_caveat: str
    fold_size_note: str


class SeedResult(BaseModel):
    seed: int
    mean_spearman: float | None
    std_spearman_across_folds: float | None


class SeedVarianceReport(BaseModel):
    status: str
    message: str
    k: int
    seeds: list[int]
    per_seed: list[SeedResult] = Field(default_factory=list)
    across_seed_mean_spearman: float | None
    across_seed_std_spearman: float | None  # NEVER dropped — seed-level stability
    small_sample_caveat: str
    fold_size_note: str


_CANNOT_RUN = (
    "CANNOT RUN — GROUND TRUTH NOT YET COLLECTED (Phase 5.1 still AWAITING REAL "
    "RATER INPUT). Refusing to produce fold/variance results on synthetic or "
    "substitute data."
)


def _cannot_run_kfold(
    k: int, seed: int, message: str = _CANNOT_RUN, n: int = 0
) -> KFoldReport:
    return KFoldReport(
        status="cannot_run",
        message=message,
        k=k,
        seed=seed,
        n=n,
        stratified_all_folds=False,
        fold_size_min=0,
        fold_size_max=0,
        mean_spearman=None,
        std_spearman=None,
        n_folds_with_spearman=0,
        small_sample_caveat=small_sample_caveat(n),
        fold_size_note="No folds — insufficient reconciled pairs.",
    )


def _reconciled(dataset: GroundTruthDataset) -> list[GroundTruthPair]:
    return [
        p
        for p in dataset.pairs
        if p.status == "reconciled" and p.reconciled_score is not None
    ]


def run_kfold_validation(
    scorer: FoldScorer,
    ground_truth_dataset: GroundTruthDataset,
    k: int = DEFAULT_K,
    seed: int = 42,
) -> KFoldReport:
    """k-fold CV of ``scorer`` on the ground truth. Reports per-fold + mean/std
    Spearman. REFUSES on unready ground truth. Deterministic given (dataset, k,
    seed)."""
    if not is_ground_truth_ready(ground_truth_dataset):
        return _cannot_run_kfold(k, seed)
    n_reconciled = len(_reconciled(ground_truth_dataset))
    if n_reconciled < 2:
        return _cannot_run_kfold(
            k,
            seed,
            message=(
                f"INSUFFICIENT DATA — k-fold needs >= 2 reconciled pairs; got "
                f"{n_reconciled}. Collect more ratings before cross-validating."
            ),
            n=n_reconciled,
        )

    pairs_by_id = {p.pair_id: p for p in ground_truth_dataset.pairs}
    folds = generate_kfolds(ground_truth_dataset, k, seed)

    per_fold: list[FoldMetrics] = []
    spearmans: list[float] = []
    for split in folds:
        train = [pairs_by_id[i] for i in split.train_pair_ids]
        test = [pairs_by_id[i] for i in split.test_pair_ids]
        scorer.fit(train)
        preds = scorer.predict(test)
        report = evaluate(preds, GroundTruthDataset(pairs=test))
        corr = report.spearman.get("correlation")
        prec = report.precision_at_5.get("precision")
        per_fold.append(
            FoldMetrics(
                fold_index=split.fold_index,
                n_test=report.n,
                spearman=corr if isinstance(corr, int | float) else None,
                precision_at_5=prec if isinstance(prec, int | float) else None,
                stratified=split.stratified,
                test_case_type_counts=split.test_case_type_counts,
            )
        )
        if isinstance(corr, int | float):
            spearmans.append(float(corr))

    agg = _aggregate(spearmans)
    sizes = [len(f.test_pair_ids) for f in folds]
    n = sum(sizes)
    fmin, fmax = min(sizes), max(sizes)
    return KFoldReport(
        status="completed",
        message="k-fold complete. std_spearman is the stability headline; a small "
        "mean with a large std means the point estimate is not trustworthy.",
        k=k,
        seed=seed,
        n=n,
        stratified_all_folds=all(f.stratified for f in folds),
        fold_size_min=fmin,
        fold_size_max=fmax,
        mean_spearman=agg["mean"] if isinstance(agg["mean"], int | float) else None,
        std_spearman=agg["std"] if isinstance(agg["std"], int | float) else None,
        n_folds_with_spearman=int(agg["n"] or 0),
        per_fold=per_fold,
        small_sample_caveat=small_sample_caveat(n),
        fold_size_note=(
            f"test folds contained {fmin}-{fmax} pair(s) each (k={k}, n={n}) — "
            f"'cross-validated' does not mean large; each fold's evaluation is tiny."
        ),
    )


def run_seed_variance_test(
    scorer: FoldScorer,
    ground_truth_dataset: GroundTruthDataset,
    k: int = DEFAULT_K,
    seed_list: tuple[int, ...] = DEFAULT_SEEDS,
) -> SeedVarianceReport:
    """Repeat k-fold across a FIXED seed list; report fold-level spread within each
    seed AND seed-level spread across seeds (the two-layer variance view)."""
    n_reconciled = len(_reconciled(ground_truth_dataset))
    if not is_ground_truth_ready(ground_truth_dataset) or n_reconciled < 2:
        msg = (
            _CANNOT_RUN
            if n_reconciled == 0
            else (
                f"INSUFFICIENT DATA — needs >= 2 reconciled pairs; got {n_reconciled}."
            )
        )
        return SeedVarianceReport(
            status="cannot_run",
            message=msg,
            k=k,
            seeds=list(seed_list),
            across_seed_mean_spearman=None,
            across_seed_std_spearman=None,
            small_sample_caveat=small_sample_caveat(n_reconciled),
            fold_size_note="No folds — insufficient reconciled pairs.",
        )

    per_seed: list[SeedResult] = []
    seed_means: list[float] = []
    fold_size_note = ""
    for seed in seed_list:
        kf = run_kfold_validation(scorer, ground_truth_dataset, k, seed)
        fold_size_note = kf.fold_size_note
        per_seed.append(
            SeedResult(
                seed=seed,
                mean_spearman=kf.mean_spearman,
                std_spearman_across_folds=kf.std_spearman,
            )
        )
        if kf.mean_spearman is not None:
            seed_means.append(kf.mean_spearman)

    across = _aggregate(seed_means)
    n_pairs = len(
        [
            p
            for p in ground_truth_dataset.pairs
            if p.status == "reconciled" and p.reconciled_score is not None
        ]
    )
    return SeedVarianceReport(
        status="completed",
        message="Seed-variance complete. across_seed_std_spearman shows how much the "
        "mean itself moves with the random split — small = stable, large = the "
        "headline number is split-luck.",
        k=k,
        seeds=list(seed_list),
        per_seed=per_seed,
        across_seed_mean_spearman=(
            across["mean"] if isinstance(across["mean"], int | float) else None
        ),
        across_seed_std_spearman=(
            across["std"] if isinstance(across["std"], int | float) else None
        ),
        small_sample_caveat=small_sample_caveat(n_pairs),
        fold_size_note=fold_size_note,
    )

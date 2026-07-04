"""Grid-Search Weight Tuning (Phase 6.4).

Systematically searches over ensemble weights for tfidf, embedding, skill overlap,
and experience match to optimize Spearman rank correlation against reconciled
ground truth.
"""

from __future__ import annotations

import difflib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from app.schemas.scoring import ConfidenceLevel, FeatureVector, ScoreResult
from app.services.evaluation.ablation_study import PairResolver, is_ground_truth_ready
from app.services.evaluation.ground_truth_schema import (
    GroundTruthDataset,
    GroundTruthPair,
)
from app.services.evaluation.kfold_stability import (
    DEFAULT_K,
    DEFAULT_SEEDS,
    FoldScorer,
    run_kfold_validation,
)
from app.services.orchestration.agent_orchestrator import OrchestratorTools
from app.services.scoring.feature_engineering import extract_feature_vector

logger = logging.getLogger(__name__)

# Canonical keys utilized in the PRD §8.2 deterministic ensemble formula
ENSEMBLE_KEYS = ["tfidf_score", "embedding_score", "skill_overlap_pct", "exp_match"]

# ============================ DISCREPANCY RESOLUTION ========================
# Interpretation (a) is selected:
# edu_match is intentionally excluded from the PRD §8.2 deterministic ensemble formula,
# as the formula specified in the locked PRD contains exactly 4 terms:
#   final_score = w1*tfidf + w2*embedding + w3*skill_overlap + w4*experience.
# While edu_match is fully utilized in Phase 6.2's trained ML models, it is not
# part of the deterministic ensemble weight tuning.


# ============================ ADAPTER FOLD SCORER ===========================


class WeightedEnsembleFoldScorer(FoldScorer):
    """Adapts a specific set of weights to the FoldScorer protocol."""

    def __init__(
        self,
        weights: dict[str, float],
        resolver: PairResolver,
        tools: OrchestratorTools,
    ) -> None:
        self.weights = weights
        self.resolver = resolver
        self.tools = tools

    def fit(self, train_pairs: list[GroundTruthPair]) -> None:
        """Deterministic model — fitting is a no-op."""
        pass

    def predict(self, test_pairs: list[GroundTruthPair]) -> list[ScoreResult]:
        preds: list[ScoreResult] = []
        for pair in test_pairs:
            resume, jd = self.resolver.resolve(pair.resume_id, pair.jd_id)
            vector = extract_feature_vector(resume, jd, self.tools)

            # Compute the weighted sum score (bounds: [0.0, 1.0])
            score = (
                self.weights.get("tfidf_score", 0.0) * vector.tfidf_score
                + self.weights.get("embedding_score", 0.0) * vector.embedding_score
                + self.weights.get("skill_overlap_pct", 0.0) * vector.skill_overlap_pct
                + self.weights.get("exp_match", 0.0) * vector.exp_match
            )

            # Scale to 0-100 and round
            final_score = min(100, max(0, round(score * 100.0)))

            preds.append(
                ScoreResult(
                    resume_id=pair.resume_id,
                    jd_id=pair.jd_id,
                    final_score=final_score,
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
                    pipeline_version="ensemble-grid",
                )
            )
        return preds


# ============================ SIMPLEX GRID GENERATION ========================


def generate_simplex_grid(step: float = 0.05) -> list[dict[str, float]]:
    """Generates a discrete simplex grid of weights summing to 1.0.

    Uses an integer-based stars-and-bars formulation to avoid floating point drift.
    """
    n_steps = int(round(1.0 / step))
    grid: list[dict[str, float]] = []

    for i1 in range(n_steps + 1):
        for i2 in range(n_steps - i1 + 1):
            for i3 in range(n_steps - i1 - i2 + 1):
                i4 = n_steps - i1 - i2 - i3
                w1 = round(i1 * step, 4)
                w2 = round(i2 * step, 4)
                w3 = round(i3 * step, 4)
                w4 = round(i4 * step, 4)

                # Ensure strict sum-to-1 constraint check
                if np.isclose(w1 + w2 + w3 + w4, 1.0):
                    grid.append(
                        {
                            "tfidf_score": w1,
                            "embedding_score": w2,
                            "skill_overlap_pct": w3,
                            "exp_match": w4,
                        }
                    )
    return grid


# ============================ SCHEMA & REPORTS ==============================


class WeightCombinationResult(BaseModel):
    weights: dict[str, float]
    mean_spearman: float | None
    std_spearman: float | None


class GridSearchReport(BaseModel):
    """The optimization report summarizing search landscape and selected winner."""

    status: Literal["completed", "cannot_run"]
    message: str
    best_weights: dict[str, float] | None = None
    best_mean_spearman: float | None = None
    best_std_spearman: float | None = None
    landscape_flatness: float | None = Field(
        None, description="Standard deviation of best-performing weights within margin."
    )
    near_best_count: int | None = Field(
        None,
        description="Number of combinations within 0.02 of the best Spearman score.",
    )
    results: list[WeightCombinationResult] = Field(default_factory=list)


# ============================ GRID SEARCH RUNNER ============================


def run_grid_search(
    ground_truth_dataset: GroundTruthDataset,
    resolver: PairResolver,
    tools: OrchestratorTools,
    step: float = 0.05,
    k: int = DEFAULT_K,
    seed_list: tuple[int, ...] = DEFAULT_SEEDS,
    margin: float = 0.02,
) -> GridSearchReport:
    """Executes discrete grid search across all candidate weight vectors.

    Refuses immediately if ground truth dataset is not ready.
    """
    if not is_ground_truth_ready(ground_truth_dataset):
        return GridSearchReport(
            status="cannot_run",
            message="CANNOT RUN — GROUND TRUTH NOT YET COLLECTED.",
        )

    grid = generate_simplex_grid(step)
    results: list[WeightCombinationResult] = []

    for weights in grid:
        scorer = WeightedEnsembleFoldScorer(weights, resolver, tools)

        # Evaluate across folds using seed_list[0]
        kfold_report = run_kfold_validation(
            scorer, ground_truth_dataset, k, seed_list[0]
        )

        results.append(
            WeightCombinationResult(
                weights=weights,
                mean_spearman=kfold_report.mean_spearman,
                std_spearman=kfold_report.std_spearman,
            )
        )

    # 1. Selection Criterion (Maximize Mean Spearman, break ties by Min Std
    # Spearman, then lexicographically)
    valid_results = [r for r in results if r.mean_spearman is not None]
    if not valid_results:
        return GridSearchReport(
            status="cannot_run",
            message=(
                "Grid search executed, but no valid Spearman correlations "
                "could be computed."
            ),
        )

    # Find the best combination
    def sorting_key(
        r: WeightCombinationResult,
    ) -> tuple[float, float, list[float]]:
        # Sort best first: highest mean (negative for min-sorting),
        # lowest std dev, lexicographical weights
        m = r.mean_spearman if r.mean_spearman is not None else -2.0
        s = r.std_spearman if r.std_spearman is not None else 999.0
        w_vals = [r.weights[k] for k in ENSEMBLE_KEYS]
        return (-m, s, w_vals)

    sorted_results = sorted(valid_results, key=sorting_key)
    best_res = sorted_results[0]
    best_mean = best_res.mean_spearman
    best_std = best_res.std_spearman

    # 2. Analyze Landscape Flatness (near-best region within margin)
    near_best = [
        r
        for r in valid_results
        if best_mean is not None and abs(best_mean - r.mean_spearman) <= margin
    ]

    # Compute standard deviation of weights in the near-best region
    # to measure landscape flatness
    weight_stds: list[float] = []
    for key in ENSEMBLE_KEYS:
        vals = [r.weights[key] for r in near_best]
        weight_stds.append(float(np.std(vals)) if len(vals) > 0 else 0.0)
    avg_weight_flatness = float(np.mean(weight_stds)) if weight_stds else 0.0

    return GridSearchReport(
        status="completed",
        message="Simplex grid search complete.",
        best_weights=best_res.weights,
        best_mean_spearman=best_mean,
        best_std_spearman=best_std,
        landscape_flatness=round(avg_weight_flatness, 6),
        near_best_count=len(near_best),
        results=results,
    )


# ============================ PROVISIONAL REPLACEMENT ========================


def apply_tuned_weights(
    agent_orchestrator_path: str | Path,
    best_weights: dict[str, float],
    gt_version: str,
    mean_spearman: float,
    std_spearman: float,
) -> str:
    """Surgically patches agent_orchestrator.py with the selected tuned weights.

    Returns the unified diff string of the changes made.
    """
    path = Path(agent_orchestrator_path)
    if not path.exists():
        raise FileNotFoundError(f"Target orchestrator path does not exist: {path}")

    content = path.read_text(encoding="utf-8")

    # Generate the target replacement string
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    weight_str_block = (
        f"# STEP 5 ensemble weights (PRD §8.2 formula:\n"
        f"#   final_score = w1*tfidf + w2*embedding + w3*skill_overlap + w4*experience).\n"  # noqa: E501
        f"# TUNED — via Phase 6.4 grid search on {timestamp} against Ground Truth {gt_version}.\n"  # noqa: E501
        f"# Verification metrics: Mean Spearman = {mean_spearman:.6f}, Std Spearman = {std_spearman:.6f}.\n"  # noqa: E501
        f"PROVISIONAL_WEIGHTS: dict[str, float] = {{\n"  # noqa: E501
        f'    "tfidf_score": {best_weights["tfidf_score"]:.4f},\n'
        f'    "embedding_score": {best_weights["embedding_score"]:.4f},\n'
        f'    "skill_overlap_pct": {best_weights["skill_overlap_pct"]:.4f},\n'
        f'    "exp_match": {best_weights["exp_match"]:.4f},\n'
        f"}}"
    )

    # Find the target block in agent_orchestrator.py
    import re

    pattern = r"(# STEP 5 ensemble weights \(PRD §8\.2 formula:.*?#   final_score = w1\*tfidf \+ w2\*embedding \+ w3\*skill_overlap \+ w4\*experience\)\..*?PROVISIONAL_WEIGHTS: dict\[str, float\] = \{.*?\})"  # noqa: E501
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(
            "Could not locate PROVISIONAL_WEIGHTS block in agent_orchestrator.py"
        )

    old_block = match.group(1)

    new_content = content.replace(old_block, weight_str_block, 1)

    # Write back the patched content
    path.write_text(new_content, encoding="utf-8")

    # Generate Unified Diff
    diff = list(
        difflib.unified_diff(
            content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=str(path.name),
            tofile=str(path.name),
        )
    )

    return "".join(diff)

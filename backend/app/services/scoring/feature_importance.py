"""Feature Importance Extraction (Phase 6.3).

Extracts, structures, and validates feature importance scores across the five
canonical features for Logistic Regression, Random Forest, and XGBoost models.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

from app.services.evaluation.ablation_study import PairResolver, is_ground_truth_ready
from app.services.evaluation.ground_truth_schema import (
    GroundTruthDataset,
)
from app.services.evaluation.kfold_stability import DEFAULT_SEEDS
from app.services.orchestration.agent_orchestrator import OrchestratorTools
from app.services.scoring.model_training import (
    LogisticRegressionRegressor,
    MLModelFoldScorer,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)

logger = logging.getLogger(__name__)

# Canonical order of features to maintain consistency with the aperture-bloom contract
FEATURE_ORDER = [
    "tfidf_score",
    "embedding_score",
    "skill_overlap_pct",
    "exp_match",
    "edu_match",
]


# ============================ SCHEMA & REPORTS ==============================


class ImportanceReport(BaseModel):
    """The unified feature importance breakdown for a single model."""

    feature_importances: dict[str, float] = Field(
        ..., description="Normalized feature importances summing to 1.0."
    )
    raw_importances: dict[str, float] = Field(
        ..., description="Raw, unnormalized coefficient or gain values."
    )
    method: str = Field(
        ..., description="The name of the importance computation method used."
    )


class FeatureStabilityStats(BaseModel):
    mean_importance: float
    std_importance: float


class ImportanceStabilityReport(BaseModel):
    """Report analyzing the stability of importances across multiple seeds."""

    status: Literal["completed", "cannot_run"]
    message: str
    feature_stability: dict[str, FeatureStabilityStats] = Field(default_factory=dict)
    raw_runs: list[dict[str, float]] = Field(default_factory=list)


class ModelComparisonEntry(BaseModel):
    tfidf_score: float
    embedding_score: float
    skill_overlap_pct: float
    exp_match: float
    edu_match: float


class CrossModelComparisonReport(BaseModel):
    """Report comparing normalized feature importances side-by-side across models."""

    status: Literal["completed", "cannot_run"]
    message: str
    comparisons: dict[str, ModelComparisonEntry] = Field(default_factory=dict)


# ============================ EXTRACTION FUNCTIONS ==========================


def extract_logreg_importance(
    trained_logreg_model: LogisticRegressionRegressor,
) -> ImportanceReport:
    """Extract feature importance from a LogisticRegressionRegressor.

    Method Used:
        Standardized Coefficient Magnitudes.

    Justification:
        Because all five canonical features are pre-normalized to the identical
        range of [0.0, 1.0] at the feature engineering layer (Phase 6.1), the
        input space is already scale-standardized. Therefore, the raw coefficient
        magnitudes of the fit Logistic Regression model are directly comparable
        as relative importance indicators without additional scaling.

    Bias/Limitations:
        Logistic regression assumes a linear relationship in log-odds.
        Multi-collinearity (e.g., between TF-IDF and embedding similarity) can
        cause coefficient values to shift or split significance between them.
    """
    if trained_logreg_model.is_constant:
        # Return flat importance if the model is in constant fallback mode
        flat_val = 1.0 / len(FEATURE_ORDER)
        return ImportanceReport(
            feature_importances={f: flat_val for f in FEATURE_ORDER},
            raw_importances={f: 0.0 for f in FEATURE_ORDER},
            method="constant_fallback",
        )

    # coef_[0] has shape (n_features,) for binary classification
    coefs = trained_logreg_model.model.coef_[0]
    abs_coefs = np.abs(coefs)

    total_abs = np.sum(abs_coefs)
    normalized = (
        abs_coefs / total_abs
        if total_abs > 0
        else np.ones(len(abs_coefs)) / len(abs_coefs)
    )

    return ImportanceReport(
        feature_importances={
            f: float(normalized[i]) for i, f in enumerate(FEATURE_ORDER)
        },
        raw_importances={f: float(coefs[i]) for i, f in enumerate(FEATURE_ORDER)},
        method="standardized_coefficients",
    )


def extract_rf_importance(
    trained_rf_model: RandomForestRegressor,
    X_val: np.ndarray | list[list[float]] | None = None,
    y_val: np.ndarray | list[float] | None = None,
) -> ImportanceReport:
    """Extract feature importance from a RandomForestRegressor.

    Method Used:
        Gini Importance (Mean Decrease in Impurity) by default, or Permutation
        Importance if validation data (X_val, y_val) is supplied.

    Justification:
        MDI is computationally efficient and built directly into the model.
        However, it can be biased towards features with high cardinality or
        overfitting. If validation data is supplied, Permutation Importance is
        used to measure actual generalization impact.

    Bias/Limitations:
        Gini importance can show optimistic bias on training data.
        Permutation importance on small validation sets (e.g. n=5-10 in folds)
        can suffer from high variance.
    """
    if X_val is not None and y_val is not None:
        X_arr = np.array(X_val)
        y_arr = np.array(y_val)
        result = permutation_importance(
            trained_rf_model, X_arr, y_arr, n_repeats=5, random_state=42
        )
        raw_vals = result.importances_mean
        method = "permutation_importance"
    else:
        raw_vals = trained_rf_model.feature_importances_
        method = "gini_importance"

    abs_vals = np.abs(raw_vals)
    total_imp = np.sum(abs_vals)
    normalized = (
        abs_vals / total_imp
        if total_imp > 0
        else np.ones(len(abs_vals)) / len(abs_vals)
    )

    return ImportanceReport(
        feature_importances={
            f: float(normalized[i]) for i, f in enumerate(FEATURE_ORDER)
        },
        raw_importances={f: float(raw_vals[i]) for i, f in enumerate(FEATURE_ORDER)},
        method=method,
    )


def extract_xgboost_importance(
    trained_xgboost_model: XGBRegressor,
) -> ImportanceReport:
    """Extract feature importance from an XGBRegressor.

    Method Used:
        Gain (fractional contribution of each feature to the model's split decisions).

    Justification:
        Gain is the most mathematically robust native tree-based importance type
        in gradient boosting, directly reflecting the improvement in accuracy
        brought by each feature split.

    Bias/Limitations:
        Tree split metrics are greedy and can favor correlated variables
        arbitrarily. Like MDI, it can overstate importance on small datasets
        due to overfitting.
    """
    # XGBoost exposes gain importance directly through feature_importances_
    raw_vals = trained_xgboost_model.feature_importances_

    abs_vals = np.abs(raw_vals)
    total_imp = np.sum(abs_vals)
    normalized = (
        abs_vals / total_imp
        if total_imp > 0
        else np.ones(len(abs_vals)) / len(abs_vals)
    )

    return ImportanceReport(
        feature_importances={
            f: float(normalized[i]) for i, f in enumerate(FEATURE_ORDER)
        },
        raw_importances={f: float(raw_vals[i]) for i, f in enumerate(FEATURE_ORDER)},
        method="gain_importance",
    )


# ============================ STABILITY CHECK ===============================


def compute_importance_stability(
    model_type: str,
    ground_truth_dataset: GroundTruthDataset,
    resolver: PairResolver,
    tools: OrchestratorTools,
    seed_list: tuple[int, ...] = DEFAULT_SEEDS,
) -> ImportanceStabilityReport:
    """Analyze the stability of feature importances across different seeds.

    Refuses immediately if ground truth dataset is not ready.
    """
    if not is_ground_truth_ready(ground_truth_dataset):
        return ImportanceStabilityReport(
            status="cannot_run",
            message="CANNOT RUN — GROUND TRUTH NOT YET COLLECTED.",
        )

    reconciled_pairs = [
        p
        for p in ground_truth_dataset.pairs
        if p.status == "reconciled" and p.reconciled_score is not None
    ]

    # Fit and extract X, y
    scorer = MLModelFoldScorer(model_type, resolver, tools)
    X, y = scorer._extract_xy(reconciled_pairs)

    raw_runs: list[dict[str, float]] = []

    for _seed in seed_list:
        if model_type == "logistic":
            model = train_logistic_regression(X, y)
            report = extract_logreg_importance(model)
        elif model_type == "random_forest":
            model = train_random_forest(X, y)
            report = extract_rf_importance(model)
        elif model_type == "xgboost":
            model = train_xgboost(X, y)
            report = extract_xgboost_importance(model)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        raw_runs.append(report.feature_importances)

    feature_stability: dict[str, FeatureStabilityStats] = {}
    for feature in FEATURE_ORDER:
        vals = [run[feature] for run in raw_runs]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
        feature_stability[feature] = FeatureStabilityStats(
            mean_importance=round(mean_val, 6),
            std_importance=round(std_val, 6),
        )

    return ImportanceStabilityReport(
        status="completed",
        message=f"Importance stability evaluated across {len(seed_list)} seeds.",
        feature_stability=feature_stability,
        raw_runs=raw_runs,
    )


# ============================ COMPARISON VIEW ===============================


def compare_model_importances(
    ground_truth_dataset: GroundTruthDataset,
    resolver: PairResolver,
    tools: OrchestratorTools,
) -> CrossModelComparisonReport:
    """Generates a side-by-side comparison of normalized importances across models."""
    if not is_ground_truth_ready(ground_truth_dataset):
        return CrossModelComparisonReport(
            status="cannot_run",
            message="CANNOT RUN — GROUND TRUTH NOT YET COLLECTED.",
        )

    reconciled_pairs = [
        p
        for p in ground_truth_dataset.pairs
        if p.status == "reconciled" and p.reconciled_score is not None
    ]

    scorer = MLModelFoldScorer("logistic", resolver, tools)
    X, y = scorer._extract_xy(reconciled_pairs)

    comparisons: dict[str, ModelComparisonEntry] = {}

    # Logistic
    lr_model = train_logistic_regression(X, y)
    lr_rep = extract_logreg_importance(lr_model)
    comparisons["logistic"] = ModelComparisonEntry(**lr_rep.feature_importances)

    # Random Forest
    rf_model = train_random_forest(X, y)
    rf_rep = extract_rf_importance(rf_model)
    comparisons["random_forest"] = ModelComparisonEntry(**rf_rep.feature_importances)

    # XGBoost
    xgb_model = train_xgboost(X, y)
    xgb_rep = extract_xgboost_importance(xgb_model)
    comparisons["xgboost"] = ModelComparisonEntry(**xgb_rep.feature_importances)

    return CrossModelComparisonReport(
        status="completed",
        message="Cross-model feature importance comparison complete.",
        comparisons=comparisons,
    )


# ============================ SUMMARY GENERATOR =============================


def generate_deterministic_importance_summary(
    report: ImportanceReport, model_type: str
) -> str:
    """Generate a deterministic, template-based summary of feature importances.

    NO LLMs. Uses plain string templating to remain traceable to the numbers.
    """
    sorted_features = sorted(
        report.feature_importances.items(), key=lambda x: (-x[1], x[0])
    )
    top_f, top_val = sorted_features[0]
    runner_f, runner_val = sorted_features[1]

    return (
        f"For {model_type}, the top driving feature was {top_f} "
        f"(normalized importance: {top_val:.4f}), followed by {runner_f} "
        f"(normalized importance: {runner_val:.4f})."
    )

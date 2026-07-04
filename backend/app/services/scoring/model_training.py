"""Trained ML Re-Ranker — Model Training Pipeline (Phase 6.2).

Trains and compares the three PRD-named candidate models (Logistic Regression,
Random Forest, XGBoost) on reconciled ground truth data using k-fold cross-
validation and seed-variance stability metrics.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor

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
    SeedVarianceReport,
    run_kfold_validation,
    run_seed_variance_test,
)
from app.services.orchestration.agent_orchestrator import OrchestratorTools
from app.services.scoring.feature_engineering import extract_feature_vector

logger = logging.getLogger(__name__)

# Base output directory for saved model artifacts.
DEFAULT_MODEL_DIR = Path("data/processed/models")

# ============================ WRAPPER CLASS =================================


class LogisticRegressionRegressor:
    """Wrapper that trains a LogisticRegression classifier on binarized targets.

    Predicts continuous scores using predict_proba scaled to 0-100.
    Satisfies identically-interfaced regressor requirements.
    """

    def __init__(self, threshold: float = 60.0, **kwargs) -> None:
        self.threshold = threshold
        self.model = LogisticRegression(**kwargs)
        self.is_constant = False
        self.constant_pred = 0

    def fit(
        self, X: np.ndarray | list[list[float]], y: np.ndarray | list[float]
    ) -> LogisticRegressionRegressor:
        X_arr = np.array(X)
        y_arr = np.array(y)
        # Binarize labels: 1 if >= threshold else 0
        y_bin = (y_arr >= self.threshold).astype(int)

        # Handle the case of zero variance / single class in small-data folds
        unique_classes = np.unique(y_bin)
        if len(unique_classes) < 2:
            self.is_constant = True
            self.constant_pred = (
                int(unique_classes[0]) if len(unique_classes) > 0 else 0
            )
            logger.warning(
                "LogisticRegression: Only one class %s present in training slice. "
                "Fitting skipped; using constant predictions.",
                self.constant_pred,
            )
        else:
            self.is_constant = False
            self.model.fit(X_arr, y_bin)

        return self

    def predict(self, X: np.ndarray | list[list[float]]) -> np.ndarray:
        X_arr = np.array(X)
        if self.is_constant:
            return np.full(len(X_arr), float(self.constant_pred * 100.0))

        # Output the probability of the positive class mapped back to [0.0, 100.0]
        probs = self.model.predict_proba(X_arr)[:, 1]
        return probs * 100.0


# ============================ STAGE FOLD ADAPTER ============================


class MLModelFoldScorer(FoldScorer):
    """Adapts scikit-learn or XGBoost regressors to the FoldScorer protocol."""

    def __init__(
        self,
        model_type: str,
        resolver: PairResolver,
        tools: OrchestratorTools,
    ) -> None:
        self.model_type = model_type
        self.resolver = resolver
        self.tools = tools
        self.model = self._init_model()

    def _init_model(self) -> any:
        if self.model_type == "logistic":
            return train_logistic_regression(np.zeros((2, 5)), [0.0, 100.0])
        elif self.model_type == "random_forest":
            return train_random_forest(np.zeros((2, 5)), [0.0, 100.0])
        elif self.model_type == "xgboost":
            return train_xgboost(np.zeros((2, 5)), [0.0, 100.0])
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _extract_xy(
        self, pairs: list[GroundTruthPair]
    ) -> tuple[list[list[float]], list[float]]:
        X: list[list[float]] = []
        y: list[float] = []
        for pair in pairs:
            resume, jd = self.resolver.resolve(pair.resume_id, pair.jd_id)
            vector = extract_feature_vector(resume, jd, self.tools)
            X.append(
                [
                    vector.tfidf_score,
                    vector.embedding_score,
                    vector.skill_overlap_pct,
                    vector.exp_match,
                    vector.edu_match,
                ]
            )
            # reconciled_score is guaranteed non-None in reconciled pairs
            y.append(
                pair.reconciled_score if pair.reconciled_score is not None else 0.0
            )
        return X, y

    def fit(self, train_pairs: list[GroundTruthPair]) -> None:
        X, y = self._extract_xy(train_pairs)
        self.model.fit(X, y)

    def predict(self, test_pairs: list[GroundTruthPair]) -> list[ScoreResult]:
        preds: list[ScoreResult] = []
        X, _ = self._extract_xy(test_pairs)
        if not X:
            return preds

        pred_scores = self.model.predict(X)
        for pair, score in zip(test_pairs, pred_scores, strict=True):
            preds.append(
                ScoreResult(
                    resume_id=pair.resume_id,
                    jd_id=pair.jd_id,
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
                    pipeline_version=f"ml-{self.model_type}",
                )
            )
        return preds


# ============================ TRAINING FUNCTIONS ============================


def train_logistic_regression(
    X: np.ndarray | list[list[float]], y: np.ndarray | list[float]
) -> LogisticRegressionRegressor:
    """Train a LogisticRegressionRegressor (binarized classification target).

    Default Hyperparameters:
        - penalty="l2": standard ridge regularization to prevent overfitting on
          highly-correlated text features.
        - C=1.0: baseline regularization strength.
        - solver="lbfgs": robust default optimizer.
        - max_iter=1000: ensures convergence on tiny data folds.
    """
    model = LogisticRegressionRegressor(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=1000
    )
    model.fit(X, y)
    return model


def train_random_forest(
    X: np.ndarray | list[list[float]], y: np.ndarray | list[float]
) -> RandomForestRegressor:
    """Train a RandomForestRegressor (continuous regression target).

    Default Hyperparameters:
        - n_estimators=100: stable number of ensemble trees.
        - max_depth=5: shallow tree depth to limit variance / overfitting on
          small-sample size.
        - min_samples_split=2: standard minimum samples split.
        - random_state=42: fixed seed for determinism.
    """
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=2,
        random_state=42,
    )
    model.fit(X, y)
    return model


def train_xgboost(
    X: np.ndarray | list[list[float]], y: np.ndarray | list[float]
) -> XGBRegressor:
    """Train an XGBRegressor (continuous regression target).

    Default Hyperparameters:
        - n_estimators=100: baseline trees.
        - max_depth=3: highly capped tree depth to prevent quick overfitting on
          sparse features or small datasets.
        - learning_rate=0.1: standard slow learning rate.
        - random_state=42: fixed seed.
        - objective="reg:squarederror": standard continuous MSE target.
    """
    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X, y)
    return model


# ============================ SCHEMA & REPORTS ==============================


class ModelCVResult(BaseModel):
    model_type: str
    mean_spearman: float | None
    std_spearman: float | None
    seed_variance: SeedVarianceReport | None


class TrainedModelReport(BaseModel):
    status: Literal["completed", "cannot_run"]
    message: str
    ground_truth_n: int
    results: list[ModelCVResult] = Field(default_factory=list)


# ============================ PIPELINE ENTRY POINT ===========================

CANNOT_TRAIN_MESSAGE = (
    "CANNOT TRAIN — GROUND TRUTH NOT YET COLLECTED. Phase 5.1's dataset has no "
    "reconciled pairs (still AWAITING REAL RATER INPUT). Refusing to train model "
    "artifacts on synthetic or placeholder data."
)


def train_and_validate(
    ground_truth_dataset: GroundTruthDataset,
    resolver: PairResolver,
    tools: OrchestratorTools,
    k: int = DEFAULT_K,
    seed_list: tuple[int, ...] = DEFAULT_SEEDS,
) -> TrainedModelReport:
    """Cross-validates the three candidate models.

    Refuses immediately if ground truth is unready.
    Uses Phase 5.4 stability machinery and produces a comparative report.
    """
    if not is_ground_truth_ready(ground_truth_dataset):
        return TrainedModelReport(
            status="cannot_run",
            message=CANNOT_TRAIN_MESSAGE,
            ground_truth_n=0,
        )

    reconciled_pairs = [
        p
        for p in ground_truth_dataset.pairs
        if p.status == "reconciled" and p.reconciled_score is not None
    ]
    n_pairs = len(reconciled_pairs)

    results: list[ModelCVResult] = []
    for model_type in ("logistic", "random_forest", "xgboost"):
        scorer = MLModelFoldScorer(model_type, resolver, tools)

        # 1. Run KFold to get mean/std Spearman on the first seed
        kfold_report = run_kfold_validation(
            scorer, ground_truth_dataset, k, seed_list[0]
        )

        # 2. Run SeedVariance to analyze split sensitivity across multiple seeds
        seed_report = run_seed_variance_test(scorer, ground_truth_dataset, k, seed_list)

        results.append(
            ModelCVResult(
                model_type=model_type,
                mean_spearman=kfold_report.mean_spearman,
                std_spearman=kfold_report.std_spearman,
                seed_variance=seed_report,
            )
        )

    return TrainedModelReport(
        status="completed",
        message="Model comparative evaluation complete.",
        ground_truth_n=n_pairs,
        results=results,
    )


# ============================ ARTIFACT PERSISTENCE ===========================


def persist_final_models(
    ground_truth_dataset: GroundTruthDataset,
    resolver: PairResolver,
    tools: OrchestratorTools,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> list[str]:
    """Train on the FULL ground-truth dataset and persist the three models.

    Refuses if ground-truth dataset is unready.
    Returns a list of saved file paths.
    """
    if not is_ground_truth_ready(ground_truth_dataset):
        logger.warning("persist_final_models: Ground truth not ready. Refusing.")
        return []

    reconciled_pairs = [
        p
        for p in ground_truth_dataset.pairs
        if p.status == "reconciled" and p.reconciled_score is not None
    ]

    # Extract X, y on full dataset
    scorer = MLModelFoldScorer("logistic", resolver, tools)
    X, y = scorer._extract_xy(reconciled_pairs)

    out_path = Path(model_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []

    timestamp = datetime.now(UTC).isoformat()
    feature_names = [
        "tfidf_score",
        "embedding_score",
        "skill_overlap_pct",
        "exp_match",
        "edu_match",
    ]

    # 1. Logistic Regression
    lr_model = train_logistic_regression(X, y)
    lr_stem = out_path / "v5-full-ml-logistic-v1"
    joblib.dump(lr_model, f"{lr_stem}.joblib")
    saved_paths.append(f"{lr_stem}.joblib")

    # 2. Random Forest
    rf_model = train_random_forest(X, y)
    rf_stem = out_path / "v5-full-ml-rf-v1"
    joblib.dump(rf_model, f"{rf_stem}.joblib")
    saved_paths.append(f"{rf_stem}.joblib")

    # 3. XGBoost
    xgb_model = train_xgboost(X, y)
    xgb_stem = out_path / "v5-full-ml-xgb-v1"
    joblib.dump(xgb_model, f"{xgb_stem}.joblib")
    saved_paths.append(f"{xgb_stem}.joblib")

    # Save provenance metadata for all models
    metadata = {
        "dataset_version": ground_truth_dataset.version,
        "n_samples": len(reconciled_pairs),
        "features": feature_names,
        "timestamp": timestamp,
        "models": {
            "logistic": {
                "artifact_path": "v5-full-ml-logistic-v1.joblib",
                "hyperparameters": {
                    "penalty": "l2",
                    "C": 1.0,
                    "solver": "lbfgs",
                    "threshold": 60.0,
                },
            },
            "random_forest": {
                "artifact_path": "v5-full-ml-rf-v1.joblib",
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "min_samples_split": 2,
                    "random_state": 42,
                },
            },
            "xgboost": {
                "artifact_path": "v5-full-ml-xgb-v1.joblib",
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 3,
                    "learning_rate": 0.1,
                    "random_state": 42,
                },
            },
        },
    }

    meta_file = out_path / "metadata.json"
    meta_file.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    saved_paths.append(str(meta_file))

    return saved_paths

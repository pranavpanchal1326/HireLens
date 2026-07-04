"""Unit tests for the Trained ML Re-Ranker Pipeline (Phase 6.2)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from app.schemas.parsing import ParsedJobDescription, ParsedResume
from app.services.evaluation.ground_truth_schema import (
    GroundTruthDataset,
    GroundTruthPair,
    RaterScore,
)
from app.services.orchestration.agent_orchestrator import OrchestratorTools
from app.services.scoring.model_training import (
    LogisticRegressionRegressor,
    MLModelFoldScorer,
    persist_final_models,
    train_and_validate,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)

# ============================ STUB SCORERS & MATCHERS ========================


class StubTFIDFScorer:
    def score(self, resume_text: str, jd_text: str) -> float:
        return 0.8


class StubCachedEmbeddingScorer:
    def score(
        self, resume_id: str, resume_text: str, jd_id: str, jd_text: str
    ) -> float:
        return 0.9


class StubSkillMatcher:
    def match_resume_to_jd(self, *args, **kwargs) -> tuple[float, list, list]:
        return 0.7, [], []


class StubExperienceMatcher:
    def match(self, resume: ParsedResume, jd: ParsedJobDescription) -> float:
        years = resume.total_years_experience
        if years is None:
            return 0.0
        return min(years / 10.0, 1.0)


class StubHybridScorer:
    def __init__(
        self, tfidf: StubTFIDFScorer, embedding: StubCachedEmbeddingScorer
    ) -> None:
        self.tfidf_scorer = tfidf
        self.cached_embedding_scorer = embedding


class StubPairResolver:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def resolve(
        self, resume_id: str, jd_id: str
    ) -> tuple[ParsedResume, ParsedJobDescription]:
        try:
            # Extract number from e.g. "r_3"
            idx = int(resume_id.split("_")[1])
        except (IndexError, ValueError):
            idx = 2

        resume = ParsedResume(
            raw_text=f"stub resume {idx}",
            skills=["Python"],
            experience=[],
            education=[],
            total_years_experience=float(idx * 2.0),
            contact_info_present=False,
            parsing_confidence=1.0,
            pipeline_version="parser-v1",
        )
        jd = ParsedJobDescription(
            raw_text="stub jd",
            required_skills=["Python"],
            preferred_skills=[],
            required_years_experience=10.0,
            required_education_level=None,
            parsing_confidence=1.0,
            pipeline_version="parser-v1",
        )
        return resume, jd


# ============================ HELPERS =======================================


def _resume() -> ParsedResume:
    return ParsedResume(
        raw_text="stub resume",
        skills=["Python"],
        experience=[],
        education=[],
        total_years_experience=2.0,
        contact_info_present=False,
        parsing_confidence=1.0,
        pipeline_version="parser-v1",
    )


def _jd() -> ParsedJobDescription:
    return ParsedJobDescription(
        raw_text="stub jd",
        required_skills=["Python"],
        preferred_skills=[],
        required_years_experience=3.0,
        required_education_level=None,
        parsing_confidence=1.0,
        pipeline_version="parser-v1",
    )


def _tools() -> OrchestratorTools:
    return OrchestratorTools(
        hybrid_scorer=StubHybridScorer(
            StubTFIDFScorer(), StubCachedEmbeddingScorer()
        ),  # type: ignore[arg-type]
        skill_matcher=StubSkillMatcher(),  # type: ignore[arg-type]
        taxonomy_entries=[],
        case_store=None,  # type: ignore[arg-type]
        experience_matcher=StubExperienceMatcher(),  # type: ignore[arg-type]
    )


def _unready_dataset() -> GroundTruthDataset:
    return GroundTruthDataset(
        version="gt-v1",
        pairs=[],
        n_raters=None,
        notes="AWAITING REAL RATER INPUT",
    )


def _ready_dataset() -> GroundTruthDataset:
    pairs = []
    # Create 6 pairs to satisfy k=5 fold requirement (needs >= k reconciled pairs)
    for i in range(6):
        pairs.append(
            GroundTruthPair(
                pair_id=f"pair_{i}",
                resume_id=f"r_{i}",
                jd_id=f"j_{i}",
                case_type="clear_fit" if i % 2 == 0 else "ambiguous",
                rater_scores=[
                    RaterScore(rater_id="h1", score=75.0, justification="ok")
                ],
                reconciled_score=80.0 if i % 2 == 0 else 40.0,
                status="reconciled",
            )
        )
    return GroundTruthDataset(
        version="gt-v1",
        pairs=pairs,
        n_raters=1,
    )


# ============================ TEST CASES =====================================


def test_readiness_check_refusal() -> None:
    dataset = _unready_dataset()
    resolver = StubPairResolver(_resume(), _jd())
    tools = _tools()
    report = train_and_validate(dataset, resolver, tools)
    assert report.status == "cannot_run"
    assert "GROUND TRUTH NOT YET COLLECTED" in report.message


def test_persist_final_models_refusal(tmp_path: Path) -> None:
    dataset = _unready_dataset()
    resolver = StubPairResolver(_resume(), _jd())
    tools = _tools()
    saved = persist_final_models(dataset, resolver, tools, model_dir=tmp_path)
    assert not saved
    assert len(list(tmp_path.glob("*"))) == 0


def test_logistic_regression_fit_predict() -> None:
    # 1. Standard two-class case
    X = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.9, 0.8, 0.7, 0.6, 0.5]]
    y = [20.0, 80.0]
    model = train_logistic_regression(X, y)
    assert isinstance(model, LogisticRegressionRegressor)
    assert not model.is_constant

    preds = model.predict(X)
    assert len(preds) == 2
    assert (
        preds[0] < preds[1]
    )  # Higher feature vector should predict higher probability/score

    # 2. Imbalance case (only one class present in training slice)
    y_imbalance = [80.0, 90.0]  # Both binarize to 1 (since >= 60)
    model_imbalance = train_logistic_regression(X, y_imbalance)
    assert model_imbalance.is_constant
    preds_imbalance = model_imbalance.predict(X)
    assert np.allclose(preds_imbalance, 100.0)


def test_random_forest_fit_predict() -> None:
    X = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.9, 0.8, 0.7, 0.6, 0.5]]
    y = [20.0, 80.0]
    model = train_random_forest(X, y)
    preds = model.predict(X)
    assert len(preds) == 2
    assert preds[0] < preds[1]


def test_xgboost_fit_predict() -> None:
    X = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.9, 0.8, 0.7, 0.6, 0.5]]
    y = [20.0, 80.0]
    model = train_xgboost(X, y)
    preds = model.predict(X)
    assert len(preds) == 2
    assert preds[0] < preds[1]


def test_ml_fold_scorer_logistic() -> None:
    dataset = _ready_dataset()
    resolver = StubPairResolver(_resume(), _jd())
    tools = _tools()
    scorer = MLModelFoldScorer("logistic", resolver, tools)
    scorer.fit(dataset.pairs[:4])
    preds = scorer.predict(dataset.pairs[4:])
    assert len(preds) == 2
    for p in preds:
        assert 0 <= p.final_score <= 100


def test_ml_fold_scorer_rf() -> None:
    dataset = _ready_dataset()
    resolver = StubPairResolver(_resume(), _jd())
    tools = _tools()
    scorer = MLModelFoldScorer("random_forest", resolver, tools)
    scorer.fit(dataset.pairs[:4])
    preds = scorer.predict(dataset.pairs[4:])
    assert len(preds) == 2
    for p in preds:
        assert 0 <= p.final_score <= 100


def test_ml_fold_scorer_xgb() -> None:
    dataset = _ready_dataset()
    resolver = StubPairResolver(_resume(), _jd())
    tools = _tools()
    scorer = MLModelFoldScorer("xgboost", resolver, tools)
    scorer.fit(dataset.pairs[:4])
    preds = scorer.predict(dataset.pairs[4:])
    assert len(preds) == 2
    for p in preds:
        assert 0 <= p.final_score <= 100


def test_no_data_leakage() -> None:
    dataset = _ready_dataset()
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    pairs = dataset.pairs
    for train_idx, test_idx in kf.split(pairs):
        train_set = {pairs[i].pair_id for i in train_idx}
        test_set = {pairs[i].pair_id for i in test_idx}
        # Assert no intersection
        assert not (train_set & test_set)


def test_determinism_and_reproducibility() -> None:
    X = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.9, 0.8, 0.7, 0.6, 0.5]]
    y = [20.0, 80.0]

    # 1. Random Forest determinism
    rf1 = train_random_forest(X, y)
    rf2 = train_random_forest(X, y)
    assert np.allclose(rf1.predict(X), rf2.predict(X))

    # 2. XGBoost determinism
    xgb1 = train_xgboost(X, y)
    xgb2 = train_xgboost(X, y)
    assert np.allclose(xgb1.predict(X), xgb2.predict(X))


def test_three_way_comparison_report_structure() -> None:
    dataset = _ready_dataset()
    resolver = StubPairResolver(_resume(), _jd())
    tools = _tools()
    report = train_and_validate(dataset, resolver, tools, k=3, seed_list=(42, 101))
    assert report.status == "completed"
    assert report.ground_truth_n == 6
    assert len(report.results) == 3
    model_types = {r.model_type for r in report.results}
    assert model_types == {"logistic", "random_forest", "xgboost"}
    for result in report.results:
        assert result.mean_spearman is not None
        assert result.seed_variance is not None
        assert result.seed_variance.across_seed_mean_spearman is not None


def test_artifact_persistence_and_provenance(tmp_path: Path) -> None:
    dataset = _ready_dataset()
    resolver = StubPairResolver(_resume(), _jd())
    tools = _tools()

    saved_files = persist_final_models(dataset, resolver, tools, model_dir=tmp_path)

    # We expect 3 model files + 1 metadata file = 4 files
    assert len(saved_files) == 4

    # Verify existence on disk
    expected_files = [
        "v5-full-ml-logistic-v1.joblib",
        "v5-full-ml-rf-v1.joblib",
        "v5-full-ml-xgb-v1.joblib",
        "metadata.json",
    ]
    for filename in expected_files:
        assert (tmp_path / filename).exists()

    # Load one back and verify it functions
    loaded_rf = joblib.load(tmp_path / "v5-full-ml-rf-v1.joblib")
    preds = loaded_rf.predict([[0.5, 0.5, 0.5, 0.5, 0.5]])
    assert len(preds) == 1

    # Verify metadata JSON structure
    meta = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert meta["dataset_version"] == "gt-v1"
    assert meta["n_samples"] == 6
    assert "logistic" in meta["models"]
    assert "random_forest" in meta["models"]
    assert "xgboost" in meta["models"]
    assert meta["models"]["xgboost"]["hyperparameters"]["random_state"] == 42

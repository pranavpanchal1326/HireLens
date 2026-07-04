# ruff: noqa: E501
"""Unit and integration tests for the /metrics endpoint (Phase 7.6).

Strategy: The metrics endpoint orchestrates complex scoring pipelines (HybridRagStage,
StageFoldAdapter, k-fold CV, evaluate). To test the ENDPOINT LOGIC (readiness
detection, trend building, persisted artifact loading, schema shape) without
requiring the full scoring infrastructure, we patch the heavy internal callables
(evaluate, _run_local_kfold, scorer creation) at the module level and use
FastAPI's app.dependency_overrides for OrchestratorTools injection.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.v1.endpoints.score import get_orchestrator_tools
from app.main import app
import app.api.v1.endpoints.metrics as metrics_module
from app.services.evaluation.ground_truth_schema import (
    GroundTruthDataset,
    GroundTruthPair,
    RaterScore,
    save_dataset,
)

client = TestClient(app, raise_server_exceptions=False)


# ============================ STUB EVALUATION REPORT =========================

def _stub_evaluation_report():
    """Return a SimpleNamespace mimicking EvaluationReport for patching evaluate()."""
    return SimpleNamespace(
        n=5,
        small_sample_caveat="PROOF-OF-CONCEPT SCALE: n=5 reconciled pair(s). Metrics are indicative only — interpret with k-fold cross-validation (Phase 5.4) and wide confidence intervals; NOT production-validated (PRD §7.3).",
        spearman={"correlation": 0.75, "p_value": 0.05, "n": 5},
        precision_at_5={"precision": 0.8, "k": 5, "k_effective": 5, "n_total": 5},
        precision_at_10={"precision": 0.6, "k": 10, "k_effective": 5, "n_total": 5},
        ndcg={"ndcg": 0.9, "k": None, "n_total": 5},
        classification=None,
        per_case_type={
            "clear_fit": {"n": 2, "spearman": {"correlation": 0.8, "p_value": 0.1, "n": 2}, "precision_at_5": {"precision": 1.0, "k": 5, "k_effective": 2, "n_total": 2}},
            "ambiguous": {"n": 3, "spearman": {"correlation": 0.7, "p_value": 0.2, "n": 3}, "precision_at_5": {"precision": 0.67, "k": 5, "k_effective": 3, "n_total": 3}},
        },
        n_predictions_unmatched=0,
    )


def _stub_kfold_stats():
    """Return a tuple of 3 MetricStat-like objects for patching _run_local_kfold()."""
    MetricStat = metrics_module.MetricStat
    return (
        MetricStat(mean=0.72, std=0.08, sample_size=5),
        MetricStat(mean=0.80, std=0.05, sample_size=5),
        MetricStat(mean=0.60, std=0.07, sample_size=5),
    )


# ============================ MOCK TOOLS =====================================

class StubHybridScorer:
    """Stub that satisfies attribute access patterns used by HybridRagStage."""
    def compute_hybrid_score(self, *args, **kwargs):
        from app.schemas.scoring import ConfidenceLevel, FeatureVector, ScoreResult
        return ScoreResult(
            resume_id="r-0", jd_id="j-0", final_score=75,
            feature_vector=FeatureVector(tfidf_score=0.7, embedding_score=0.8, skill_overlap_pct=0.7, exp_match=0.8, edu_match=0.5),
            scoring_confidence=0.6, confidence_level=ConfidenceLevel.MEDIUM, parsing_confidence=0.9, pipeline_version="v3-hybrid",
        )


class StubSkillMatcher:
    def match_resume_to_jd(self, *args, **kwargs):
        return (0.7, [], [])


class StubExperienceMatcher:
    def match(self, *args, **kwargs):
        return 0.8


class StubCaseStore:
    def build_case_embedding(self, *args, **kwargs):
        return None
    def retrieve_similar_cases(self, *args, **kwargs):
        return []
    def calibration_check(self, *args, **kwargs):
        return SimpleNamespace(is_outlier=False, deviation=0.0, similar_case_ids=[], similar_case_scores=[], low_sample_warning=False)


_MOCK_TOOLS = None
def _get_mock_tools():
    global _MOCK_TOOLS
    if _MOCK_TOOLS is None:
        from app.services.orchestration.agent_orchestrator import OrchestratorTools
        _MOCK_TOOLS = OrchestratorTools(
            hybrid_scorer=StubHybridScorer(),  # type: ignore[arg-type]
            skill_matcher=StubSkillMatcher(),  # type: ignore[arg-type]
            taxonomy_entries=[],
            case_store=StubCaseStore(),  # type: ignore[arg-type]
            experience_matcher=StubExperienceMatcher(),  # type: ignore[arg-type]
        )
    return _MOCK_TOOLS


# ============================ FIXTURES =======================================

@pytest.fixture(autouse=True)
def mock_metrics_paths(tmp_path: Path):
    """Redirect all persistence file paths to tmp_path for test isolation."""
    orig_dataset = metrics_module.DATASET_PATH
    orig_history = metrics_module.METRICS_HISTORY_PATH
    orig_importance = metrics_module.FEATURE_IMPORTANCE_PATH
    orig_grid = metrics_module.GRID_SEARCH_PATH

    metrics_module.DATASET_PATH = tmp_path / "ground_truth_dataset.json"
    metrics_module.METRICS_HISTORY_PATH = tmp_path / "metrics_history.json"
    metrics_module.FEATURE_IMPORTANCE_PATH = tmp_path / "feature_importance.json"
    metrics_module.GRID_SEARCH_PATH = tmp_path / "grid_search.json"

    yield

    metrics_module.DATASET_PATH = orig_dataset
    metrics_module.METRICS_HISTORY_PATH = orig_history
    metrics_module.FEATURE_IMPORTANCE_PATH = orig_importance
    metrics_module.GRID_SEARCH_PATH = orig_grid


@pytest.fixture(autouse=True)
def override_tools():
    """Use FastAPI dependency overrides (not patch) for OrchestratorTools DI."""
    app.dependency_overrides[get_orchestrator_tools] = _get_mock_tools
    yield
    app.dependency_overrides.pop(get_orchestrator_tools, None)


@pytest.fixture
def populated_dataset() -> GroundTruthDataset:
    """A ground-truth dataset with 5 reconciled pairs for testing."""
    pairs = []
    for i in range(5):
        pairs.append(
            GroundTruthPair(
                pair_id=f"gt-{i:02d}",
                resume_id=f"r-{i}",
                jd_id=f"j-{i}",
                case_type="ambiguous" if i % 2 == 0 else "clear_fit",
                rater_scores=[
                    RaterScore(rater_id="rater-1", score=80.0 + i, justification="Valid justification text"),
                    RaterScore(rater_id="rater-2", score=85.0 + i, justification="Valid justification text"),
                    RaterScore(rater_id="rater-3", score=90.0 + i, justification="Valid justification text"),
                ],
                reconciled_score=85.0 + i,
                inter_rater_range=10.0,
                divergence_flag=False,
                status="reconciled",
            )
        )
    return GroundTruthDataset(
        version="gt-v1",
        pairs=pairs,
        n_raters=3,
        overall_inter_rater_agreement=0.95,
        notes="Populated test dataset",
    )


def _patch_scoring_internals(maturity_status: str = "provisional"):
    """Context manager stack that patches evaluate, _run_local_kfold, maturity, and scorer creation."""
    from contextlib import ExitStack
    stack = ExitStack()

    # 1. Pipeline maturity
    mock_mat = stack.enter_context(
        patch("app.api.v1.endpoints.metrics.get_pipeline_maturity_status")
    )
    if maturity_status == "tuned":
        mock_mat.return_value = {
            "status": "tuned", "weights_status": "tuned",
            "model_status": "trained",
            "details": "System is fully calibrated and trained against ground truth.",
        }
    else:
        mock_mat.return_value = {
            "status": "provisional", "weights_status": "provisional",
            "model_status": "provisional",
            "details": "System is in a provisional/placeholder state pending ground truth calibration.",
        }

    # 2. evaluate() — returns a controlled EvaluationReport
    mock_eval = stack.enter_context(
        patch("app.api.v1.endpoints.metrics.evaluate")
    )
    mock_eval.return_value = _stub_evaluation_report()

    # 3. _run_local_kfold() — returns controlled MetricStat tuples
    mock_kfold = stack.enter_context(
        patch("app.api.v1.endpoints.metrics._run_local_kfold")
    )
    mock_kfold.return_value = _stub_kfold_stats()

    # 4. CSVGroundTruthResolver — stub resolver that returns mock parsed objects
    mock_resolver_cls = stack.enter_context(
        patch("app.api.v1.endpoints.metrics.CSVGroundTruthResolver")
    )
    mock_resolver_instance = MagicMock()
    mock_resolver_cls.return_value = mock_resolver_instance

    # 5. StageFoldAdapter / MLModelFoldScorer — stub scorer with predict()
    from app.schemas.scoring import ConfidenceLevel, FeatureVector, ScoreResult
    def _make_predictions(pairs):
        return [
            ScoreResult(
                resume_id=p.resume_id, jd_id=p.jd_id, final_score=85,
                feature_vector=FeatureVector(tfidf_score=0.7, embedding_score=0.8, skill_overlap_pct=0.7, exp_match=0.8, edu_match=0.5),
                scoring_confidence=0.6, confidence_level=ConfidenceLevel.MEDIUM, parsing_confidence=0.9, pipeline_version="v4-hybrid-rag",
            )
            for p in pairs
        ]

    mock_adapter_cls = stack.enter_context(
        patch("app.api.v1.endpoints.metrics.StageFoldAdapter")
    )
    mock_scorer = MagicMock()
    mock_scorer.predict.side_effect = _make_predictions
    mock_adapter_cls.return_value = mock_scorer

    mock_ml_cls = stack.enter_context(
        patch("app.api.v1.endpoints.metrics.MLModelFoldScorer")
    )
    mock_ml_scorer = MagicMock()
    mock_ml_scorer.predict.side_effect = _make_predictions
    mock_ml_cls.return_value = mock_ml_scorer

    # 6. HybridRagStage — stub so it doesn't try to instantiate real scorers
    mock_stage_cls = stack.enter_context(
        patch("app.api.v1.endpoints.metrics.HybridRagStage")
    )
    mock_stage_cls.return_value = MagicMock()

    return stack


# ============================ TESTS ==========================================


# 1. Unready State (no reconciled pairs)
def test_metrics_unready_state() -> None:
    if metrics_module.DATASET_PATH.exists():
        metrics_module.DATASET_PATH.unlink()

    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["readiness_state"] == "unready"
    assert "no rated pairs found" in data["status_details"]
    assert data["current_metrics"] is None
    assert data["trend"] == []


# 2. Provisional State
def test_metrics_provisional_state(populated_dataset) -> None:
    save_dataset(populated_dataset, str(metrics_module.DATASET_PATH))

    with _patch_scoring_internals("provisional"):
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["readiness_state"] == "provisional"
        assert "provisional weights" in data["status_details"]
        current = data["current_metrics"]
        assert current is not None
        assert current["n"] == 5
        assert "PROOF-OF-CONCEPT SCALE" in current["small_sample_caveat"]
        assert "mean" in current["spearman"]
        assert "std" in current["spearman"]


# 3. Tuned State
def test_metrics_tuned_state(populated_dataset) -> None:
    save_dataset(populated_dataset, str(metrics_module.DATASET_PATH))

    with _patch_scoring_internals("tuned"):
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["readiness_state"] == "tuned"
        assert "tuned weights" in data["status_details"]


# 4. Mean and Std Dev Always Paired
def test_metrics_mean_std_present(populated_dataset) -> None:
    save_dataset(populated_dataset, str(metrics_module.DATASET_PATH))

    with _patch_scoring_internals("provisional"):
        response = client.get("/api/v1/metrics")
        data = response.json()
        current = data["current_metrics"]

        assert current["spearman"]["mean"] is not None
        assert current["spearman"]["std"] is not None
        assert current["spearman"]["sample_size"] == 5

        assert current["precision_at_5"]["mean"] is not None
        assert current["precision_at_5"]["std"] is not None
        assert current["precision_at_5"]["sample_size"] == 5

        assert current["precision_at_10"]["mean"] is not None
        assert current["precision_at_10"]["std"] is not None
        assert current["precision_at_10"]["sample_size"] == 5


# 5. Read-only Guarantee (two consecutive calls produce zero side-effects)
def test_metrics_read_only_guarantee(populated_dataset) -> None:
    save_dataset(populated_dataset, str(metrics_module.DATASET_PATH))

    with _patch_scoring_internals("provisional"):
        before_mod_time = metrics_module.DATASET_PATH.stat().st_mtime

        r1 = client.get("/api/v1/metrics")
        assert r1.status_code == 200

        r2 = client.get("/api/v1/metrics")
        assert r2.status_code == 200

        assert metrics_module.DATASET_PATH.stat().st_mtime == before_mod_time

        # Responses should be structurally identical (timestamps in trend may differ
        # by milliseconds, so we compare everything except trend timestamp fields)
        d1, d2 = r1.json(), r2.json()
        assert d1["readiness_state"] == d2["readiness_state"]
        assert d1["current_metrics"] == d2["current_metrics"]
        assert d1["feature_importance"] == d2["feature_importance"]
        assert d1["grid_search"] == d2["grid_search"]


# 6. Trend from History File
def test_metrics_history_trend(populated_dataset) -> None:
    save_dataset(populated_dataset, str(metrics_module.DATASET_PATH))

    history_data = [
        {
            "pipeline_version": "v3-hybrid",
            "dataset_size": 3,
            "timestamp": "2026-07-04T12:00:00Z",
            "spearman_mean": 0.5,
            "spearman_std": 0.1,
            "precision_at_5_mean": 0.6,
            "precision_at_5_std": 0.05,
            "precision_at_10_mean": 0.4,
            "precision_at_10_std": 0.05,
        }
    ]
    metrics_module.METRICS_HISTORY_PATH.write_text(json.dumps(history_data), encoding="utf-8")

    with _patch_scoring_internals("provisional"):
        response = client.get("/api/v1/metrics")
        data = response.json()
        assert len(data["trend"]) == 1
        assert data["trend"][0]["pipeline_version"] == "v3-hybrid"
        assert data["trend"][0]["dataset_size"] == 3


# 7. Single Point Fallback when History is Missing
def test_metrics_single_point_trend_fallback(populated_dataset) -> None:
    save_dataset(populated_dataset, str(metrics_module.DATASET_PATH))

    if metrics_module.METRICS_HISTORY_PATH.exists():
        metrics_module.METRICS_HISTORY_PATH.unlink()

    with _patch_scoring_internals("provisional"):
        response = client.get("/api/v1/metrics")
        data = response.json()
        assert len(data["trend"]) == 1
        assert data["trend"][0]["pipeline_version"] == "v4-hybrid-rag"
        assert data["trend"][0]["dataset_size"] == 5


# 8. Feature Importance Loading
def test_metrics_feature_importance_loading(populated_dataset) -> None:
    save_dataset(populated_dataset, str(metrics_module.DATASET_PATH))

    importance_data = {
        "status": "completed",
        "message": "Cross-model feature importance comparison complete.",
        "comparisons": {
            "logistic": {
                "tfidf_score": 0.2, "embedding_score": 0.2,
                "skill_overlap_pct": 0.2, "exp_match": 0.2, "edu_match": 0.2,
            }
        },
    }
    metrics_module.FEATURE_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_module.FEATURE_IMPORTANCE_PATH.write_text(json.dumps(importance_data), encoding="utf-8")

    with _patch_scoring_internals("provisional"):
        response = client.get("/api/v1/metrics")
        data = response.json()
        assert data["feature_importance"] is not None
        assert "logistic" in data["feature_importance"]["comparisons"]


# 9. Grid Search Loading
def test_metrics_grid_search_loading(populated_dataset) -> None:
    save_dataset(populated_dataset, str(metrics_module.DATASET_PATH))

    grid_search_data = {
        "status": "completed",
        "message": "Simplex grid search complete.",
        "best_weights": {"tfidf_score": 0.25, "embedding_score": 0.25, "skill_overlap_pct": 0.25, "exp_match": 0.25},
        "best_mean_spearman": 0.85,
        "best_std_spearman": 0.05,
        "landscape_flatness": 0.012,
        "near_best_count": 5,
    }
    metrics_module.GRID_SEARCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_module.GRID_SEARCH_PATH.write_text(json.dumps(grid_search_data), encoding="utf-8")

    with _patch_scoring_internals("provisional"):
        response = client.get("/api/v1/metrics")
        data = response.json()
        assert data["grid_search"] is not None
        assert data["grid_search"]["landscape_flatness"] == 0.012


# 10. Honest Absence Handling
def test_metrics_honest_absence_handling(populated_dataset) -> None:
    save_dataset(populated_dataset, str(metrics_module.DATASET_PATH))

    if metrics_module.FEATURE_IMPORTANCE_PATH.exists():
        metrics_module.FEATURE_IMPORTANCE_PATH.unlink()
    if metrics_module.GRID_SEARCH_PATH.exists():
        metrics_module.GRID_SEARCH_PATH.unlink()

    with _patch_scoring_internals("provisional"):
        response = client.get("/api/v1/metrics")
        data = response.json()
        assert data["feature_importance"] is None
        assert data["grid_search"] is None


# 11. Case Type Breakdown Present
def test_metrics_case_type_breakdown(populated_dataset) -> None:
    save_dataset(populated_dataset, str(metrics_module.DATASET_PATH))

    with _patch_scoring_internals("provisional"):
        response = client.get("/api/v1/metrics")
        data = response.json()
        current = data["current_metrics"]
        assert "per_case_type" in current
        assert "clear_fit" in current["per_case_type"]
        assert "ambiguous" in current["per_case_type"]
        assert current["per_case_type"]["clear_fit"]["n"] > 0

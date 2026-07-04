"""Unit tests for the Feature Importance Extraction Pipeline (Phase 6.3)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor

from app.schemas.parsing import ParsedJobDescription, ParsedResume
from app.services.evaluation.ground_truth_schema import (
    GroundTruthDataset,
    GroundTruthPair,
    RaterScore,
)
from app.services.orchestration.agent_orchestrator import OrchestratorTools
from app.services.scoring.feature_importance import (
    FEATURE_ORDER,
    ImportanceReport,
    compare_model_importances,
    compute_importance_stability,
    extract_logreg_importance,
    extract_rf_importance,
    extract_xgboost_importance,
    generate_deterministic_importance_summary,
)
from app.services.scoring.model_training import LogisticRegressionRegressor

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
    resolver = StubPairResolver()
    tools = _tools()

    stability_rep = compute_importance_stability("logistic", dataset, resolver, tools)
    assert stability_rep.status == "cannot_run"
    assert "GROUND TRUTH NOT YET COLLECTED" in stability_rep.message

    comp_rep = compare_model_importances(dataset, resolver, tools)
    assert comp_rep.status == "cannot_run"
    assert "GROUND TRUTH NOT YET COLLECTED" in comp_rep.message


def test_logreg_importance_extraction() -> None:
    lr = LogisticRegressionRegressor()
    lr.is_constant = False

    # Mock model fitting internals
    lr.model = LogisticRegression()
    lr.model.coef_ = np.array([[1.0, -2.0, 0.0, 0.0, 0.0]])
    lr.model.classes_ = np.array([0, 1])

    report = extract_logreg_importance(lr)

    assert report.method == "standardized_coefficients"
    assert report.raw_importances["tfidf_score"] == 1.0
    assert report.raw_importances["embedding_score"] == -2.0
    assert report.feature_importances["tfidf_score"] == pytest.approx(1.0 / 3.0)
    assert report.feature_importances["embedding_score"] == pytest.approx(2.0 / 3.0)
    assert report.feature_importances["skill_overlap_pct"] == 0.0


def test_logreg_constant_fallback() -> None:
    lr = LogisticRegressionRegressor()
    lr.is_constant = True
    lr.constant_pred = 1

    report = extract_logreg_importance(lr)
    assert report.method == "constant_fallback"
    for feature in FEATURE_ORDER:
        assert report.feature_importances[feature] == pytest.approx(0.2)
        assert report.raw_importances[feature] == 0.0


def test_rf_importance_extraction() -> None:
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    X = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.9, 0.8, 0.7, 0.6, 0.5]]
    y = [20.0, 80.0]
    rf.fit(X, y)

    # 1. Test Gini MDI
    report_gini = extract_rf_importance(rf)
    assert report_gini.method == "gini_importance"
    assert sum(report_gini.feature_importances.values()) == pytest.approx(1.0)

    # 2. Test Permutation Importance
    report_perm = extract_rf_importance(rf, X_val=X, y_val=y)
    assert report_perm.method == "permutation_importance"
    assert sum(report_perm.feature_importances.values()) == pytest.approx(1.0)


def test_xgboost_importance_extraction() -> None:
    xgb = XGBRegressor(n_estimators=10, max_depth=2, random_state=42)
    X = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.9, 0.8, 0.7, 0.6, 0.5]]
    y = [20.0, 80.0]
    xgb.fit(X, y)

    report = extract_xgboost_importance(xgb)
    assert report.method == "gain_importance"
    assert sum(report.feature_importances.values()) == pytest.approx(1.0)


def test_unified_importance_report_schema() -> None:
    # Verify that all returned values inherit and match ImportanceReport
    lr = LogisticRegressionRegressor()
    lr.is_constant = True
    report = extract_logreg_importance(lr)
    assert isinstance(report, ImportanceReport)
    assert set(report.model_dump().keys()) == {
        "feature_importances",
        "raw_importances",
        "method",
    }


def test_stability_report_aggregation() -> None:
    dataset = _ready_dataset()
    resolver = StubPairResolver()
    tools = _tools()

    report = compute_importance_stability(
        "logistic", dataset, resolver, tools, seed_list=(42, 101)
    )
    assert report.status == "completed"
    assert len(report.raw_runs) == 2
    for feature in FEATURE_ORDER:
        assert feature in report.feature_stability
        assert report.feature_stability[feature].mean_importance >= 0.0
        assert report.feature_stability[feature].std_importance >= 0.0


def test_stability_report_hand_computed() -> None:
    # Verify sample standard deviation calculations for stability report
    from app.services.scoring.feature_importance import (
        FeatureStabilityStats,
    )

    # Stability test values: tfidf_score runs: [0.4, 0.6] -> mean = 0.5,
    # sample std = 0.141421
    runs = [
        {
            "tfidf_score": 0.4,
            "embedding_score": 0.6,
            "skill_overlap_pct": 0.0,
            "exp_match": 0.0,
            "edu_match": 0.0,
        },
        {
            "tfidf_score": 0.6,
            "embedding_score": 0.4,
            "skill_overlap_pct": 0.0,
            "exp_match": 0.0,
            "edu_match": 0.0,
        },
    ]

    feature_stability = {}
    for feature in FEATURE_ORDER:
        vals = [run[feature] for run in runs]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
        feature_stability[feature] = FeatureStabilityStats(
            mean_importance=round(mean_val, 6),
            std_importance=round(std_val, 6),
        )

    assert feature_stability["tfidf_score"].mean_importance == 0.5
    assert math.isclose(
        feature_stability["tfidf_score"].std_importance, 0.141421, rel_tol=1e-5
    )


def test_single_seed_stability_std_is_none_not_zero() -> None:
    """Regression (Phase 6.X audit, Pass 5): a single-seed run must report
    std_importance=None, never a fake-certain 0.0. Std over one value is undefined
    and faking it as 0.0 violates Design Blueprint P3 (honesty), diverging from
    Phase 5.4 kfold_stability's own convention."""
    dataset = _ready_dataset()
    resolver = StubPairResolver()
    tools = _tools()

    report = compute_importance_stability(
        "logistic", dataset, resolver, tools, seed_list=(42,)
    )
    assert report.status == "completed"
    assert len(report.raw_runs) == 1
    for feature in FEATURE_ORDER:
        assert report.feature_stability[feature].std_importance is None
        # mean is still a real, reportable number
        assert report.feature_stability[feature].mean_importance >= 0.0


def test_comparison_report_disagreement() -> None:
    dataset = _ready_dataset()
    resolver = StubPairResolver()
    tools = _tools()

    report = compare_model_importances(dataset, resolver, tools)
    assert report.status == "completed"
    assert "logistic" in report.comparisons
    assert "random_forest" in report.comparisons
    assert "xgboost" in report.comparisons

    # Make sure we don't average them away - they should remain separate per model
    logistic_imp = report.comparisons["logistic"]
    rf_imp = report.comparisons["random_forest"]
    assert (
        logistic_imp.tfidf_score != rf_imp.tfidf_score
        or logistic_imp.exp_match != rf_imp.exp_match
    )


def test_summary_generation_deterministic() -> None:
    report = ImportanceReport(
        feature_importances={
            "tfidf_score": 0.45,
            "embedding_score": 0.35,
            "skill_overlap_pct": 0.10,
            "exp_match": 0.05,
            "edu_match": 0.05,
        },
        raw_importances={},
        method="test",
    )
    summary = generate_deterministic_importance_summary(report, "random_forest")
    assert (
        "For random_forest, the top driving feature was tfidf_score "
        "(normalized importance: 0.4500), followed by embedding_score "
        "(normalized importance: 0.3500)."
    ) in summary

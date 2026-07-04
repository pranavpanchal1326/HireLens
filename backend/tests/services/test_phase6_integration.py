"""Phase 6.X — End-to-end integration test (audit Pass 7).

First proof that Phase 6.1 → 6.2 → 6.3 → 6.4 fit together as ONE working chain,
not four independently-correct pieces whose mutual compatibility was only assumed.

SYNTHETIC-FIXTURE-ONLY. This module never touches real/placeholder ground truth
and never persists anything that could be mistaken for a production model, weight
set, or importance ranking. The fixture pairs are explicitly labelled reconciled
so they pass the SAME readiness gate the real pipeline uses — a Pass 0 pre-flight
assertion guarantees the chain is exercised rather than short-circuited to
`cannot_run` (which would make every downstream assertion vacuous).
"""

from __future__ import annotations

import pytest

from app.schemas.parsing import ParsedJobDescription, ParsedResume
from app.schemas.scoring import FeatureVector
from app.services.evaluation.ablation_study import is_ground_truth_ready
from app.services.evaluation.ground_truth_schema import (
    GroundTruthDataset,
    GroundTruthPair,
    RaterScore,
)
from app.services.orchestration.agent_orchestrator import OrchestratorTools
from app.services.scoring.feature_engineering import extract_feature_vector
from app.services.scoring.feature_importance import (
    FEATURE_ORDER,
    compare_model_importances,
    compute_importance_stability,
)
from app.services.scoring.grid_search_tuning import ENSEMBLE_KEYS, run_grid_search
from app.services.scoring.model_training import reconciled_pairs, train_and_validate

# ============================ SYNTHETIC STUBS ================================
# Feature values vary with the candidate index so the chain sees real signal
# spread (not a constant), which lets importance/grid-search produce non-trivial
# structure to assert on.


class StubTFIDFScorer:
    def score(self, resume_text: str, jd_text: str) -> float:
        return 0.8


class StubCachedEmbeddingScorer:
    def score(
        self, resume_id: str, resume_text: str, jd_id: str, jd_text: str
    ) -> float:
        return 0.6


class StubSkillMatcher:
    def match_resume_to_jd(self, *args, **kwargs) -> tuple[float, list, list]:
        return 0.5, [], []


class StubExperienceMatcher:
    def match(self, resume: ParsedResume, jd: ParsedJobDescription) -> float:
        years = resume.total_years_experience or 0.0
        return min(years / 10.0, 1.0)


class StubHybridScorer:
    def __init__(self, tfidf: StubTFIDFScorer, emb: StubCachedEmbeddingScorer) -> None:
        self.tfidf_scorer = tfidf
        self.cached_embedding_scorer = emb


class StubPairResolver:
    def resolve(
        self, resume_id: str, jd_id: str
    ) -> tuple[ParsedResume, ParsedJobDescription]:
        try:
            idx = int(resume_id.split("_")[1])
        except (IndexError, ValueError):
            idx = 2
        resume = ParsedResume(
            raw_text=f"synthetic resume {idx}",
            skills=["Python"],
            experience=[],
            education=[],
            total_years_experience=float(idx * 2.0),
            contact_info_present=False,
            parsing_confidence=1.0,
            pipeline_version="parser-v1",
        )
        jd = ParsedJobDescription(
            raw_text="synthetic jd",
            required_skills=["Python"],
            preferred_skills=[],
            required_years_experience=10.0,
            required_education_level=None,
            parsing_confidence=1.0,
            pipeline_version="parser-v1",
        )
        return resume, jd


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


def _synthetic_reconciled_dataset(n: int = 8) -> GroundTruthDataset:
    """SYNTHETIC ground truth — explicitly reconciled so it passes the real gate."""
    pairs = [
        GroundTruthPair(
            pair_id=f"pair_{i}",
            resume_id=f"r_{i}",
            jd_id=f"j_{i}",
            case_type="clear_fit" if i % 2 == 0 else "ambiguous",
            rater_scores=[
                RaterScore(rater_id="synthetic", score=70.0, justification="fixture")
            ],
            reconciled_score=85.0 if i % 2 == 0 else 35.0,
            status="reconciled",
        )
        for i in range(n)
    ]
    return GroundTruthDataset(version="gt-synthetic", pairs=pairs, n_raters=1)


# ============================ PASS 0 — PRE-FLIGHT ============================


def test_pass0_fixture_passes_real_readiness_gate() -> None:
    """The chain is only meaningfully tested if the fixture is genuinely 'ready'
    under the SAME gate production uses. Otherwise every stage returns cannot_run
    and the rest of this module would pass vacuously."""
    dataset = _synthetic_reconciled_dataset()
    assert is_ground_truth_ready(dataset) is True
    assert len(reconciled_pairs(dataset)) == 8


# ============================ PASS 7 — FULL CHAIN ===========================


def test_phase6_end_to_end_chain() -> None:
    dataset = _synthetic_reconciled_dataset()
    resolver = StubPairResolver()
    tools = _tools()

    # --- Stage 6.1: feature extraction produces the locked 5-vector in [0,1] ---
    pairs = reconciled_pairs(dataset)
    resume, jd = resolver.resolve(pairs[0].resume_id, pairs[0].jd_id)
    fv = extract_feature_vector(resume, jd, tools)
    assert isinstance(fv, FeatureVector)
    assert list(fv.model_dump().keys()) == FEATURE_ORDER
    for value in fv.model_dump().values():
        assert 0.0 <= value <= 1.0

    # --- Stage 6.2: train + validate all three models on 6.1 features ---
    train_report = train_and_validate(dataset, resolver, tools, k=3, seed_list=(11, 42))
    assert train_report.status == "completed"
    assert train_report.ground_truth_n == 8
    assert {r.model_type for r in train_report.results} == {
        "logistic",
        "random_forest",
        "xgboost",
    }
    # Disagreement preserved, not averaged away (Pass 6).
    assert len(train_report.results) == 3
    for r in train_report.results:
        assert r.seed_variance is not None  # mean+std machinery wired through

    # --- Stage 6.3: importance extraction + stability on fixture-trained models ---
    for model_type in ("logistic", "random_forest", "xgboost"):
        stab = compute_importance_stability(
            model_type, dataset, resolver, tools, seed_list=(11, 42)
        )
        assert stab.status == "completed"
        assert set(stab.feature_stability.keys()) == set(FEATURE_ORDER)
        for feat in FEATURE_ORDER:
            # 2 seeds -> std is a real float, mean is present.
            assert stab.feature_stability[feat].mean_importance >= 0.0

    comparison = compare_model_importances(dataset, resolver, tools)
    assert comparison.status == "completed"
    assert set(comparison.comparisons.keys()) == {
        "logistic",
        "random_forest",
        "xgboost",
    }

    # --- Stage 6.4: grid search reuses the same fixture feature vectors + gt ---
    grid_report = run_grid_search(
        dataset, resolver, tools, step=0.5, k=3, seed_list=(42,)
    )
    assert grid_report.status == "completed"
    assert grid_report.best_weights is not None
    # 6.4 tunes exactly the 4 ensemble features; edu_match is not a weight.
    assert set(grid_report.best_weights.keys()) == set(ENSEMBLE_KEYS)
    assert "edu_match" not in grid_report.best_weights
    # Simplex constraint: the tuned weights sum to 1.0.
    assert sum(grid_report.best_weights.values()) == pytest.approx(1.0)
    # Landscape disagreement surfaced, not collapsed (Pass 6).
    assert grid_report.near_best_count is not None
    assert len(grid_report.results) > 1


def test_phase6_chain_refuses_on_unready_ground_truth() -> None:
    """The whole chain must refuse (never fabricate) when ground truth is unready —
    the actual product-safety guarantee, exercised end to end."""
    unready = GroundTruthDataset(
        version="gt-v1", pairs=[], notes="AWAITING REAL RATER INPUT"
    )
    resolver = StubPairResolver()
    tools = _tools()

    assert is_ground_truth_ready(unready) is False
    assert train_and_validate(unready, resolver, tools).status == "cannot_run"
    assert (
        compute_importance_stability("logistic", unready, resolver, tools).status
        == "cannot_run"
    )
    assert compare_model_importances(unready, resolver, tools).status == "cannot_run"
    assert run_grid_search(unready, resolver, tools).status == "cannot_run"

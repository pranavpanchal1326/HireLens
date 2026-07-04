# ruff: noqa: E501
"""Unit tests for the Grid-Search Weight Tuning Pipeline (Phase 6.4)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from app.schemas.parsing import ParsedJobDescription, ParsedResume
from app.services.evaluation.ground_truth_schema import (
    GroundTruthDataset,
    GroundTruthPair,
    RaterScore,
)
from app.services.orchestration.agent_orchestrator import OrchestratorTools
from app.services.scoring.grid_search_tuning import (
    WeightCombinationResult,
    WeightedEnsembleFoldScorer,
    apply_tuned_weights,
    generate_simplex_grid,
    run_grid_search,
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

    report = run_grid_search(dataset, resolver, tools)
    assert report.status == "cannot_run"
    assert "GROUND TRUTH NOT YET COLLECTED" in report.message


def test_simplex_grid_generation_step_0_5() -> None:
    # step=0.5 -> N=2 steps. Combos = stars-and-bars(2, 4) = (2+4-1) choose (4-1) = 5 choose 3 = 10.
    grid = generate_simplex_grid(step=0.5)
    assert len(grid) == 10
    for combo in grid:
        assert len(combo) == 4
        assert math.isclose(sum(combo.values()), 1.0, rel_tol=1e-6)


def test_simplex_grid_generation_step_0_2() -> None:
    # step=0.2 -> N=5 steps. Combos = stars-and-bars(5, 4) = (5+4-1) choose (4-1) = 8 choose 3 = 56.
    grid = generate_simplex_grid(step=0.2)
    assert len(grid) == 56
    for combo in grid:
        assert math.isclose(sum(combo.values()), 1.0, rel_tol=1e-6)


def test_weighted_ensemble_fold_scorer_predict() -> None:
    weights = {
        "tfidf_score": 0.25,
        "embedding_score": 0.25,
        "skill_overlap_pct": 0.25,
        "exp_match": 0.25,
    }
    resolver = StubPairResolver()
    tools = _tools()
    scorer = WeightedEnsembleFoldScorer(weights, resolver, tools)

    pairs = [
        GroundTruthPair(
            pair_id="p1",
            resume_id="r_2",
            jd_id="j_2",
            case_type="clear_fit",
            reconciled_score=80.0,
            status="reconciled",
        )
    ]
    preds = scorer.predict(pairs)

    assert len(preds) == 1
    pred = preds[0]
    # tfidf: 0.8, embedding: 0.9, skill_overlap: 0.7, exp: min(4.0/10.0, 1.0) = 0.4
    # weighted: 0.25*(0.8 + 0.9 + 0.7 + 0.4) = 0.25*(2.8) = 0.7
    # 0.7 * 100 = 70.0
    assert pred.final_score == 70


def test_selection_and_tiebreaker() -> None:
    # Mock some run_grid_search outputs directly
    results = [
        WeightCombinationResult(
            weights={
                "tfidf_score": 0.25,
                "embedding_score": 0.25,
                "skill_overlap_pct": 0.25,
                "exp_match": 0.25,
            },
            mean_spearman=0.65,
            std_spearman=0.10,
        ),
        WeightCombinationResult(
            weights={
                "tfidf_score": 0.1,
                "embedding_score": 0.3,
                "skill_overlap_pct": 0.3,
                "exp_match": 0.3,
            },
            mean_spearman=0.65,
            std_spearman=0.05,  # Tie breaker: lower std dev!
        ),
        WeightCombinationResult(
            weights={
                "tfidf_score": 0.4,
                "embedding_score": 0.2,
                "skill_overlap_pct": 0.2,
                "exp_match": 0.2,
            },
            mean_spearman=0.60,
            std_spearman=0.02,
        ),
    ]

    # Best is mean=0.65, std=0.05
    sorted_results = sorted(
        results,
        key=lambda r: (
            -r.mean_spearman if r.mean_spearman is not None else 2.0,
            r.std_spearman if r.std_spearman is not None else 999.0,
            [
                r.weights[k]
                for k in [
                    "tfidf_score",
                    "embedding_score",
                    "skill_overlap_pct",
                    "exp_match",
                ]
            ],
        ),
    )

    assert sorted_results[0].weights["tfidf_score"] == 0.1
    assert sorted_results[0].std_spearman == 0.05


def test_landscape_flatness_calculation() -> None:
    import numpy as np

    # Simulates near_best region where scores are within 0.02 of best mean (0.70)
    near_best = [
        WeightCombinationResult(
            weights={
                "tfidf_score": 0.25,
                "embedding_score": 0.25,
                "skill_overlap_pct": 0.25,
                "exp_match": 0.25,
            },
            mean_spearman=0.70,
            std_spearman=0.05,
        ),
        WeightCombinationResult(
            weights={
                "tfidf_score": 0.20,
                "embedding_score": 0.30,
                "skill_overlap_pct": 0.25,
                "exp_match": 0.25,
            },
            mean_spearman=0.69,
            std_spearman=0.05,
        ),
    ]

    # Stds of weights for each key:
    # tfidf: std([0.25, 0.20]) = 0.025
    # embedding: std([0.25, 0.30]) = 0.025
    # skill_overlap: std([0.25, 0.25]) = 0.0
    # exp: std([0.25, 0.25]) = 0.0
    # Mean of stds = 0.05 / 4 = 0.0125
    stds = []
    for key in ["tfidf_score", "embedding_score", "skill_overlap_pct", "exp_match"]:
        vals = [r.weights[key] for r in near_best]
        stds.append(float(np.std(vals)))

    avg_flatness = float(np.mean(stds))
    assert avg_flatness == pytest.approx(0.0125)


def test_grid_search_integration() -> None:
    dataset = _ready_dataset()
    resolver = StubPairResolver()
    tools = _tools()

    report = run_grid_search(dataset, resolver, tools, step=0.5, k=2)
    assert report.status == "completed"
    assert report.best_weights is not None
    assert report.best_mean_spearman is not None
    assert report.best_std_spearman is not None
    assert report.near_best_count is not None
    assert report.near_best_count > 0


def test_apply_tuned_weights_patching(tmp_path: Path) -> None:
    dummy_orchestrator = (
        "# Number of similar past cases to pull for calibration (Phase 3.4).\n"
        "CALIBRATION_TOP_K = 5\n\n"
        "# STEP 5 ensemble weights (PRD §8.2 formula:\n"
        "#   final_score = w1*tfidf + w2*embedding + w3*skill_overlap + w4*experience).\n"
        "# THESE WEIGHTS ARE PLACEHOLDERS — Phase 6.4 grid search will OVERWRITE them via\n"
        "# calibration against ground truth. Do NOT treat as final tuned values. Sum to 1.0.\n"
        "PROVISIONAL_WEIGHTS: dict[str, float] = {\n"
        '    "tfidf_score": 0.25,\n'
        '    "embedding_score": 0.25,\n'
        '    "skill_overlap_pct": 0.30,\n'
        '    "exp_match": 0.20,\n'
        "}\n\n"
        "# STEP 5: spread across the 4 weighted features beyond which we pull the final\n"
    )

    orchestrator_file = tmp_path / "agent_orchestrator.py"
    orchestrator_file.write_text(dummy_orchestrator, encoding="utf-8")

    best_weights = {
        "tfidf_score": 0.15,
        "embedding_score": 0.35,
        "skill_overlap_pct": 0.25,
        "exp_match": 0.25,
    }

    diff = apply_tuned_weights(
        orchestrator_file,
        best_weights,
        gt_version="gt-v1",
        mean_spearman=0.685412,
        std_spearman=0.043210,
    )

    assert diff != ""
    assert "PROVISIONAL_WEIGHTS: dict[str, float]" in diff
    assert '+    "tfidf_score": 0.1500,' in diff
    assert '-    "tfidf_score": 0.25,' in diff

    # Verify write-back
    updated_content = orchestrator_file.read_text(encoding="utf-8")
    assert '"tfidf_score": 0.1500,' in updated_content
    assert "TUNED — via Phase 6.4" in updated_content
    assert "Verification metrics: Mean Spearman = 0.685412" in updated_content


def test_apply_tuned_weights_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        apply_tuned_weights(
            "nonexistent_file.py",
            {
                "tfidf_score": 0.25,
                "embedding_score": 0.25,
                "skill_overlap_pct": 0.25,
                "exp_match": 0.25,
            },
            "v1",
            0.6,
            0.05,
        )


def test_apply_tuned_weights_invalid_pattern(tmp_path: Path) -> None:
    orchestrator_file = tmp_path / "agent_orchestrator.py"
    orchestrator_file.write_text("SOME DUMMY PYTHON CODE", encoding="utf-8")
    with pytest.raises(ValueError):
        apply_tuned_weights(
            orchestrator_file,
            {
                "tfidf_score": 0.25,
                "embedding_score": 0.25,
                "skill_overlap_pct": 0.25,
                "exp_match": 0.25,
            },
            "v1",
            0.6,
            0.05,
        )

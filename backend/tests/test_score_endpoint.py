# ruff: noqa: E501
"""Unit and integration tests for the /score endpoint (Phase 7.3)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api.v1.endpoints.score import get_orchestrator_tools
from app.main import app
from app.schemas.parsing import ParsedJobDescription, ParsedResume
from app.schemas.scoring import (
    ConfidenceLevel,
    FeatureVector,
    GapItem,
    ScoreResult,
    SkillMatch,
)
from app.services.orchestration.agent_orchestrator import OrchestratorTools

client = TestClient(app, raise_server_exceptions=False)


# ============================ STUB INSTANCES =================================


class StubExperienceMatcher:
    def match(self, resume: ParsedResume, jd: ParsedJobDescription) -> float:
        return 0.8


class StubSkillMatcher:
    def match_resume_to_jd(self, *args, **kwargs) -> tuple[float, list, list]:
        return (
            0.7,
            [
                SkillMatch(
                    resume_skill="a",
                    jd_skill="a",
                    match_type="exact",
                    similarity_score=1.0,
                )
            ],
            [GapItem(missing_skill="b", suggested_action="Learn b")],
        )


class StubHybridScorer:
    def compute_hybrid_score(self, *args, **kwargs) -> ScoreResult:
        return ScoreResult(
            resume_id="r1",
            jd_id="j1",
            final_score=75,
            feature_vector=FeatureVector(
                tfidf_score=0.7,
                embedding_score=0.8,
                skill_overlap_pct=0.7,
                exp_match=0.8,
                edu_match=0.0,
            ),
            scoring_confidence=0.75,
            confidence_level=ConfidenceLevel.HIGH,
            parsing_confidence=0.9,
            pipeline_version="v3-hybrid",
        )


class StubCaseStore:
    def build_case_embedding(self, *args, **kwargs):
        return None

    def retrieve_similar_cases(self, *args, **kwargs):
        return []

    def calibration_check(self, *args, **kwargs):
        class DummyCalibration:
            is_outlier = False
            deviation = 0.0
            similar_case_ids = []
            similar_case_scores = []
            low_sample_warning = False

        return DummyCalibration()


# A reusable mocked OrchestratorTools block for testing
_MOCK_TOOLS = OrchestratorTools(
    hybrid_scorer=StubHybridScorer(),  # type: ignore[arg-type]
    skill_matcher=StubSkillMatcher(),  # type: ignore[arg-type]
    taxonomy_entries=[],
    case_store=StubCaseStore(),  # type: ignore[arg-type]
    experience_matcher=StubExperienceMatcher(),  # type: ignore[arg-type]
)


@pytest.fixture(autouse=True)
def override_dependencies() -> None:
    app.dependency_overrides[get_orchestrator_tools] = lambda: _MOCK_TOOLS
    yield
    app.dependency_overrides.clear()


# ============================ TEST CASES =====================================


def test_score_success_high_confidence() -> None:
    """Verify successful score run under high-confidence conditions."""
    req_payload = {
        "parsed_resume": {
            "raw_text": "Jane Doe. Python, SQL. 5 years exp.",
            "skills": ["Python", "SQL"],
            "experience": [],
            "education": [],
            "total_years_experience": 5.0,
            "contact_info_present": True,
            "parsing_confidence": 1.0,
            "pipeline_version": "parser-v1",
        },
        "parsed_jd": {
            "raw_text": "Required: Python, SQL. 3 years exp.",
            "required_skills": ["Python", "SQL"],
            "preferred_skills": [],
            "required_years_experience": 3.0,
            "required_education_level": None,
            "parsing_confidence": 1.0,
            "pipeline_version": "parser-v1",
        },
    }

    response = client.post("/api/v1/score", json=req_payload)
    assert response.status_code == 200
    data = response.json()

    assert "score_result" in data
    assert "pipeline_maturity" in data

    score_res = data["score_result"]
    assert score_res["final_score"] == 75
    assert score_res["confidence_level"] == "high"
    assert len(score_res["matched_skills"]) == 1
    assert len(score_res["gaps"]) == 1


def test_score_success_low_confidence() -> None:
    """Verify that low-confidence scores return a 200 with low confidence label."""
    # Force low-confidence by mocking the orchestrator or return values
    with patch("app.api.v1.endpoints.score.run_orchestration") as mock_orchestrate:
        mock_orchestrate.return_value = ScoreResult(
            resume_id="r1",
            jd_id="j1",
            final_score=20,
            feature_vector=FeatureVector(
                tfidf_score=0.1,
                embedding_score=0.2,
                skill_overlap_pct=0.1,
                exp_match=0.0,
                edu_match=0.0,
            ),
            scoring_confidence=0.15,
            confidence_level=ConfidenceLevel.LOW,
            parsing_confidence=0.8,
            pipeline_version="v3-hybrid",
            confidence_reasons=["Low lexical overlap"],
        )

        req_payload = {
            "raw_resume_text": "Jane Doe.",
            "raw_jd_text": "Required: Senior Accountant.",
        }

        response = client.post("/api/v1/score", json=req_payload)
        assert response.status_code == 200
        data = response.json()

        score_res = data["score_result"]
        assert score_res["confidence_level"] == "low"
        assert score_res["final_score"] == 20


def test_score_provisional_maturity_surfacing() -> None:
    """Verify that provisional maturity status is correctly surfaced in metadata."""
    # Ensure best_model.joblib is not detected to keep it provisional
    with patch("app.api.v1.endpoints.score.Path.exists", return_value=False):
        response = client.post(
            "/api/v1/score",
            json={
                "raw_resume_text": "Jane Doe.",
                "raw_jd_text": "Staff Software Engineer.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        maturity = data["pipeline_maturity"]
        assert maturity["status"] == "provisional"
        assert maturity["weights_status"] == "provisional"
        assert maturity["model_status"] == "provisional"


def test_score_tuned_maturity_surfacing() -> None:
    """Verify that tuned maturity status is correctly surfaced when artifacts exist."""
    # Mock Path.exists to say both best_model.joblib and # TUNED orchestrator comment exist
    with patch("app.api.v1.endpoints.score.Path.exists", return_value=True):
        with patch(
            "app.api.v1.endpoints.score.Path.read_text",
            return_value="# TUNED — via Phase 6.4 grid search",
        ):
            response = client.post(
                "/api/v1/score",
                json={
                    "raw_resume_text": "Jane Doe.",
                    "raw_jd_text": "Staff Software Engineer.",
                },
            )
            assert response.status_code == 200
            data = response.json()
            maturity = data["pipeline_maturity"]
            assert maturity["status"] == "tuned"
            assert maturity["weights_status"] == "tuned"
            assert maturity["model_status"] == "trained"


def test_score_missing_inputs() -> None:
    """Verify that missing inputs return HTTP 400 Bad Request."""
    response = client.post("/api/v1/score", json={})
    assert response.status_code == 400
    assert (
        "parsed_resume or raw_resume_text must be provided"
        in response.json()["message"]
    )


def test_score_upstream_failure_attribution() -> None:
    """Verify that upstream orchestrator crashes produce a safe 500 error instead of faking a score."""
    with patch(
        "app.api.v1.endpoints.score.run_orchestration",
        side_effect=ValueError("Skill taxonomy query failed"),
    ):
        response = client.post(
            "/api/v1/score",
            json={
                "raw_resume_text": "Jane Doe.",
                "raw_jd_text": "Staff Software Engineer.",
            },
        )
        assert response.status_code == 500
        data = response.json()
        assert data["code"] == "INTERNAL_SERVER_ERROR"
        # Safe message, traceback hidden
        assert "Skill taxonomy query failed" not in data["message"]


def test_score_timeout_error() -> None:
    """Verify that requests exceeding SCORE_TIMEOUT_SECONDS trigger an HTTP 504 Gateway Timeout."""

    def slow_orchestration(*args, **kwargs) -> ScoreResult:
        import time

        time.sleep(1.0)
        return ScoreResult(
            resume_id="r1",
            jd_id="j1",
            final_score=80,
            feature_vector=FeatureVector(
                tfidf_score=0.8,
                embedding_score=0.8,
                skill_overlap_pct=0.8,
                exp_match=0.8,
                edu_match=0.0,
            ),
            scoring_confidence=0.8,
            confidence_level=ConfidenceLevel.HIGH,
            parsing_confidence=0.9,
            pipeline_version="v3-hybrid",
        )

    with patch("app.api.v1.endpoints.score.run_orchestration", new=slow_orchestration):
        with patch("app.api.v1.endpoints.score.SCORE_TIMEOUT_SECONDS", 0.05):
            response = client.post(
                "/api/v1/score",
                json={
                    "raw_resume_text": "Jane Doe.",
                    "raw_jd_text": "Staff Software Engineer.",
                },
            )
            assert response.status_code == 504
            assert "timed out" in response.json()["message"]


def test_score_field_parity_lossless() -> None:
    """Verify that there is zero information loss between internal OrchestrationResult and external API response."""
    reasons = [
        "STEP 4: signal spread exceeds threshold",
        "STEP 5: matched experience is close to target",
    ]
    matched_skills = [
        SkillMatch(
            resume_skill="Python",
            jd_skill="Python",
            match_type="exact",
            similarity_score=1.0,
        ),
        SkillMatch(
            resume_skill="Docker",
            jd_skill="Containerization",
            match_type="semantic",
            similarity_score=0.85,
        ),
    ]
    gaps = [
        GapItem(
            missing_skill="Kubernetes",
            suggested_action="Deploy containers with Kubernetes",
        ),
    ]

    expected_result = ScoreResult(
        score_id="test-score-uuid",
        resume_id="r_99",
        jd_id="j_100",
        final_score=88,
        feature_vector=FeatureVector(
            tfidf_score=0.75,
            embedding_score=0.92,
            skill_overlap_pct=0.68,
            exp_match=0.80,
            edu_match=0.0,
        ),
        scoring_confidence=0.84,
        confidence_level=ConfidenceLevel.HIGH,
        parsing_confidence=0.95,
        matched_skills=matched_skills,
        gaps=gaps,
        confidence_reasons=reasons,
        pipeline_version="ensemble-grid-test",
    )

    with patch(
        "app.api.v1.endpoints.score.run_orchestration", return_value=expected_result
    ):
        response = client.post(
            "/api/v1/score",
            json={
                "raw_resume_text": "Jane Doe.",
                "raw_jd_text": "Staff Software Engineer.",
            },
        )

        assert response.status_code == 200
        data = response.json()
        score_res = data["score_result"]

        # Assert precise field-for-field parity
        assert score_res["score_id"] == "test-score-uuid"
        assert score_res["resume_id"] == "r_99"
        assert score_res["jd_id"] == "j_100"
        assert score_res["final_score"] == 88
        assert score_res["scoring_confidence"] == pytest.approx(0.84)
        assert score_res["confidence_level"] == "high"
        assert score_res["parsing_confidence"] == pytest.approx(0.95)
        assert score_res["pipeline_version"] == "ensemble-grid-test"

        # Check matched skills list completeness
        assert len(score_res["matched_skills"]) == 2
        assert score_res["matched_skills"][0]["resume_skill"] == "Python"
        assert score_res["matched_skills"][1]["match_type"] == "semantic"
        assert score_res["matched_skills"][1]["similarity_score"] == pytest.approx(0.85)

        # Check gaps completeness
        assert len(score_res["gaps"]) == 1
        assert score_res["gaps"][0]["missing_skill"] == "Kubernetes"
        assert "Deploy containers" in score_res["gaps"][0]["suggested_action"]

        # Check reasons completeness
        # The endpoint appends its own maturity status reason, so we check that the base ones are present
        reasons_list = score_res["confidence_reasons"]
        assert reasons[0] in reasons_list
        assert reasons[1] in reasons_list
        assert any("Pipeline maturity status" in r for r in reasons_list)

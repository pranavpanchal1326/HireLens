# ruff: noqa: E501
"""Unit and integration tests for the /rank endpoint (Phase 7.4)."""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api.v1.endpoints.rank import get_orchestrator_tools
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


# Reusable mocked tools
_MOCK_TOOLS = OrchestratorTools(
    hybrid_scorer=StubHybridScorer(),  # type: ignore[arg-type]
    skill_matcher=StubSkillMatcher(),  # type: ignore[arg-type]
    taxonomy_entries=[],
    case_store=StubCaseStore(),  # type: ignore[arg-type]
    experience_matcher=StubExperienceMatcher(),  # type: ignore[arg-type]
)


@pytest.fixture(autouse=True)
def override_dependencies() -> None:
    from app.core.auth import get_current_recruiter, RecruiterAccount
    app.dependency_overrides[get_orchestrator_tools] = lambda: _MOCK_TOOLS
    app.dependency_overrides[get_current_recruiter] = lambda: RecruiterAccount(
        account_id="company_a", recruiter_id="recruiter_one"
    )
    yield
    app.dependency_overrides.clear()


# ============================ TEST CASES =====================================


def test_rank_success_full_batch() -> None:
    """Verify that a batch of valid resumes ranks successfully in the correct order."""
    # We mock run_orchestration to return distinct scores for different candidates
    scores_map = {
        "candidate_A": 90,
        "candidate_B": 80,
        "candidate_C": 95,
    }

    def mock_orchestration_run(resume, jd, tools):
        # Infer candidate score based on raw_text content
        score = 50
        for name, val in scores_map.items():
            if name in resume.raw_text:
                score = val
        return ScoreResult(
            resume_id=resume.raw_text,
            jd_id="jd-1",
            final_score=score,
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

    req_payload = {
        "raw_jd_text": "Requirements here.",
        "resumes": [
            {"candidate_id": "A", "raw_resume_text": "candidate_A skills"},
            {"candidate_id": "B", "raw_resume_text": "candidate_B skills"},
            {"candidate_id": "C", "raw_resume_text": "candidate_C skills"},
        ],
    }

    with patch(
        "app.api.v1.endpoints.rank.run_orchestration",
        side_effect=mock_orchestration_run,
    ):
        response = client.post("/api/v1/rank", json=req_payload)
        assert response.status_code == 200
        data = response.json()

        # Overall checks
        assert data["total_submitted"] == 3
        assert data["total_successful"] == 3
        assert data["total_failed"] == 0
        assert len(data["failures"]) == 0

        # Order check: C (95) -> A (90) -> B (80)
        candidates = data["ranking_result"]["ranked_candidates"]
        assert len(candidates) == 3

        assert candidates[0]["candidate_id"] == "C"
        assert candidates[0]["rank"] == 1
        assert candidates[0]["score_result"]["final_score"] == 95

        assert candidates[1]["candidate_id"] == "A"
        assert candidates[1]["rank"] == 2
        assert candidates[1]["score_result"]["final_score"] == 90

        assert candidates[2]["candidate_id"] == "B"
        assert candidates[2]["rank"] == 3
        assert candidates[2]["score_result"]["final_score"] == 80


def test_rank_partial_failures() -> None:
    """Verify that a failing resume does not crash the entire batch (resilience check)."""

    # A mocks that fails candidate_B but passes others
    def mock_orchestration_run(resume, jd, tools):
        if "candidate_B" in resume.raw_text:
            raise ValueError("Corrupt PDF structure or parsing exception")
        return ScoreResult(
            resume_id="res",
            jd_id="jd-1",
            final_score=85,
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

    req_payload = {
        "raw_jd_text": "Requirements here.",
        "resumes": [
            {"candidate_id": "A", "raw_resume_text": "candidate_A skills"},
            {"candidate_id": "B", "raw_resume_text": "candidate_B skills"},
            {"candidate_id": "C", "raw_resume_text": "candidate_C skills"},
        ],
    }

    with patch(
        "app.api.v1.endpoints.rank.run_orchestration",
        side_effect=mock_orchestration_run,
    ):
        response = client.post("/api/v1/rank", json=req_payload)
        assert (
            response.status_code == 200
        )  # Enforced: must return 200 on partial failure
        data = response.json()

        assert data["total_submitted"] == 3
        assert data["total_successful"] == 2
        assert data["total_failed"] == 1

        # Check successful candidates list
        candidates = data["ranking_result"]["ranked_candidates"]
        assert len(candidates) == 2
        assert {c["candidate_id"] for c in candidates} == {"A", "C"}

        # Check fail-row trace and reason attribution
        failures = data["failures"]
        assert len(failures) == 1
        assert failures[0]["candidate_id"] == "B"
        assert "Corrupt PDF structure" in failures[0]["reason"]


def test_rank_jd_parsed_exactly_once() -> None:
    """Assert that the job description is parsed exactly once per request."""
    # We patch structure_job_description to track calls
    req_payload = {
        "raw_jd_text": "Need Python developer.",
        "resumes": [
            {"candidate_id": "A", "raw_resume_text": "A content"},
            {"candidate_id": "B", "raw_resume_text": "B content"},
        ],
    }

    with patch("app.api.v1.endpoints.rank.structure_job_description") as mock_parse_jd:
        mock_parse_jd.return_value = ParsedJobDescription(
            raw_text="Need Python developer.",
            required_skills=["Python"],
            preferred_skills=[],
            required_years_experience=2.0,
            required_education_level=None,
            parsing_confidence=1.0,
            pipeline_version="parser-v1",
        )

        response = client.post("/api/v1/rank", json=req_payload)
        assert response.status_code == 200
        # Confirms: parsed exactly 1 time, not per-resume (2 times)
        assert mock_parse_jd.call_count == 1


def test_rank_maturity_called_exactly_once() -> None:
    """Assert that get_pipeline_maturity_status is called exactly once per request."""
    req_payload = {
        "raw_jd_text": "Need Python developer.",
        "resumes": [
            {"candidate_id": "A", "raw_resume_text": "A content"},
            {"candidate_id": "B", "raw_resume_text": "B content"},
        ],
    }

    with patch(
        "app.api.v1.endpoints.rank.get_pipeline_maturity_status"
    ) as mock_maturity:
        mock_maturity.return_value = {
            "status": "provisional",
            "weights_status": "provisional",
            "model_status": "provisional",
            "details": "Details",
        }

        response = client.post("/api/v1/rank", json=req_payload)
        assert response.status_code == 200
        # Confirms: called exactly 1 time, not per-resume
        assert mock_maturity.call_count == 1


def test_rank_oversized_batch_rejection() -> None:
    """Assert that requests containing more than MAX_BATCH_SIZE resumes are rejected with HTTP 422."""
    resumes_payload = [
        {"candidate_id": str(i), "raw_resume_text": "skills"} for i in range(51)
    ]
    req_payload = {
        "raw_jd_text": "JD text",
        "resumes": resumes_payload,
    }

    response = client.post("/api/v1/rank", json=req_payload)
    assert response.status_code == 422
    assert "at most 50 items" in response.json()["details"][0]["msg"]


def test_rank_tied_score_deterministic_ordering() -> None:
    """Verify deterministic tie-breaking: higher score first, then candidate_id ascending."""

    # We mock scoring to return identical scores of 80 for all candidates
    def mock_orchestration_run(resume, jd, tools):
        return ScoreResult(
            resume_id="res",
            jd_id="jd-1",
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

    # Shuffled input order
    candidates_list = ["zebra", "apple", "monkey", "banana"]
    random.shuffle(candidates_list)
    resumes_payload = [
        {"candidate_id": name, "raw_resume_text": "skills"} for name in candidates_list
    ]

    req_payload = {
        "raw_jd_text": "JD text",
        "resumes": resumes_payload,
    }

    with patch(
        "app.api.v1.endpoints.rank.run_orchestration",
        side_effect=mock_orchestration_run,
    ):
        # Run it multiple times to ensure absolute stability
        for _ in range(5):
            response = client.post("/api/v1/rank", json=req_payload)
            assert response.status_code == 200
            data = response.json()
            candidates = data["ranking_result"]["ranked_candidates"]

            # Sorted order MUST be: apple (1) -> banana (2) -> monkey (3) -> zebra (4)
            assert candidates[0]["candidate_id"] == "apple"
            assert candidates[0]["rank"] == 1

            assert candidates[1]["candidate_id"] == "banana"
            assert candidates[1]["rank"] == 2

            assert candidates[2]["candidate_id"] == "monkey"
            assert candidates[2]["rank"] == 3

            assert candidates[3]["candidate_id"] == "zebra"
            assert candidates[3]["rank"] == 4


def test_rank_field_parity_lossless() -> None:
    """Verify row-by-row field-for-field parity with zero information loss."""
    reasons_a = ["REASON A1", "REASON A2"]
    reasons_b = ["REASON B1"]

    mock_results = {
        "A": ScoreResult(
            score_id="uuid-a",
            resume_id="r-a",
            jd_id="j1",
            final_score=90,
            feature_vector=FeatureVector(
                tfidf_score=0.9,
                embedding_score=0.9,
                skill_overlap_pct=0.9,
                exp_match=0.9,
                edu_match=0.0,
            ),
            scoring_confidence=0.9,
            confidence_level=ConfidenceLevel.HIGH,
            parsing_confidence=0.9,
            matched_skills=[
                SkillMatch(
                    resume_skill="Python",
                    jd_skill="Python",
                    match_type="exact",
                    similarity_score=1.0,
                )
            ],
            gaps=[GapItem(missing_skill="SQL", suggested_action="Learn SQL")],
            confidence_reasons=reasons_a,
            pipeline_version="ensemble-v1",
        ),
        "B": ScoreResult(
            score_id="uuid-b",
            resume_id="r-b",
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
            confidence_level=ConfidenceLevel.MEDIUM,
            parsing_confidence=0.8,
            matched_skills=[],
            gaps=[],
            confidence_reasons=reasons_b,
            pipeline_version="ensemble-v1",
        ),
    }

    def mock_orchestration_run(resume, jd, tools):
        for candidate_id, res in mock_results.items():
            if f"candidate_{candidate_id}" in resume.raw_text:
                return res
        raise ValueError("Unknown candidate")

    req_payload = {
        "raw_jd_text": "JD text",
        "resumes": [
            {"candidate_id": "A", "raw_resume_text": "candidate_A skills"},
            {"candidate_id": "B", "raw_resume_text": "candidate_B skills"},
        ],
    }

    with patch(
        "app.api.v1.endpoints.rank.run_orchestration",
        side_effect=mock_orchestration_run,
    ):
        response = client.post("/api/v1/rank", json=req_payload)
        assert response.status_code == 200
        data = response.json()
        candidates = data["ranking_result"]["ranked_candidates"]

        # Check candidate A
        c_a = next(c for c in candidates if c["candidate_id"] == "A")
        assert c_a["score_result"]["score_id"] == "uuid-a"
        assert c_a["score_result"]["final_score"] == 90
        assert c_a["score_result"]["confidence_level"] == "high"
        assert len(c_a["score_result"]["matched_skills"]) == 1
        assert c_a["score_result"]["matched_skills"][0]["resume_skill"] == "Python"
        assert len(c_a["score_result"]["gaps"]) == 1
        assert c_a["score_result"]["gaps"][0]["missing_skill"] == "SQL"

        # Verify reasons completeness (excluding appended maturity status)
        assert reasons_a[0] in c_a["score_result"]["confidence_reasons"]
        assert reasons_a[1] in c_a["score_result"]["confidence_reasons"]

        # Check candidate B
        c_b = next(c for c in candidates if c["candidate_id"] == "B")
        assert c_b["score_result"]["score_id"] == "uuid-b"
        assert c_b["score_result"]["final_score"] == 80
        assert c_b["score_result"]["confidence_level"] == "medium"
        assert len(c_b["score_result"]["matched_skills"]) == 0
        assert len(c_b["score_result"]["gaps"]) == 0
        assert reasons_b[0] in c_b["score_result"]["confidence_reasons"]


def test_rank_empty_resumes_list_rejection() -> None:
    """Verify that requests containing an empty resumes list are rejected with HTTP 422."""
    req_payload = {
        "raw_jd_text": "JD text",
        "resumes": [],
    }
    response = client.post("/api/v1/rank", json=req_payload)
    assert response.status_code == 422
    assert "at least 1 item" in response.json()["details"][0]["msg"]


def test_rank_missing_jd_fails() -> None:
    """Verify that missing both parsed_jd and raw_jd_text triggers a 400 Bad Request."""
    req_payload = {
        "resumes": [{"candidate_id": "A", "raw_resume_text": "resume skills"}]
    }
    response = client.post("/api/v1/rank", json=req_payload)
    assert response.status_code == 400
    assert (
        "Either parsed_jd or raw_jd_text must be provided" in response.json()["message"]
    )


def test_rank_single_candidate_batch() -> None:
    """Verify that a batch with exactly N=1 candidate works properly."""

    def mock_orchestration_run(resume, jd, tools):
        return ScoreResult(
            resume_id="res",
            jd_id="jd-1",
            final_score=85,
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

    req_payload = {
        "raw_jd_text": "JD text",
        "resumes": [{"candidate_id": "single", "raw_resume_text": "resume skills"}],
    }

    with patch(
        "app.api.v1.endpoints.rank.run_orchestration",
        side_effect=mock_orchestration_run,
    ):
        response = client.post("/api/v1/rank", json=req_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_submitted"] == 1
        assert data["total_successful"] == 1
        assert (
            data["ranking_result"]["ranked_candidates"][0]["candidate_id"] == "single"
        )
        assert data["ranking_result"]["ranked_candidates"][0]["rank"] == 1


def test_rank_preparsed_resumes_success() -> None:
    """Verify that using pre-parsed resumes in payloads runs successfully."""

    def mock_orchestration_run(resume, jd, tools):
        assert resume.total_years_experience == 5.0
        return ScoreResult(
            resume_id="res",
            jd_id="jd-1",
            final_score=90,
            feature_vector=FeatureVector(
                tfidf_score=0.9,
                embedding_score=0.9,
                skill_overlap_pct=0.9,
                exp_match=0.9,
                edu_match=0.0,
            ),
            scoring_confidence=0.9,
            confidence_level=ConfidenceLevel.HIGH,
            parsing_confidence=1.0,
            pipeline_version="v3-hybrid",
        )

    req_payload = {
        "raw_jd_text": "JD text",
        "resumes": [
            {
                "candidate_id": "preparsed",
                "parsed_resume": {
                    "raw_text": "Jane Doe.",
                    "skills": ["Python"],
                    "experience": [],
                    "education": [],
                    "total_years_experience": 5.0,
                    "contact_info_present": True,
                    "parsing_confidence": 1.0,
                    "pipeline_version": "parser-v1",
                },
            }
        ],
    }

    with patch(
        "app.api.v1.endpoints.rank.run_orchestration",
        side_effect=mock_orchestration_run,
    ):
        response = client.post("/api/v1/rank", json=req_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_successful"] == 1
        assert (
            data["ranking_result"]["ranked_candidates"][0]["candidate_id"]
            == "preparsed"
        )


def test_rank_all_candidates_failed() -> None:
    """Verify that a batch where all resumes fail returns 200 with total_failed == total_submitted."""

    def mock_orchestration_run(resume, jd, tools):
        raise ValueError("Parser error")

    req_payload = {
        "raw_jd_text": "JD text",
        "resumes": [
            {"candidate_id": "A", "raw_resume_text": "resume skills"},
            {"candidate_id": "B", "raw_resume_text": "resume skills"},
        ],
    }

    with patch(
        "app.api.v1.endpoints.rank.run_orchestration",
        side_effect=mock_orchestration_run,
    ):
        response = client.post("/api/v1/rank", json=req_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_submitted"] == 2
        assert data["total_successful"] == 0
        assert data["total_failed"] == 2
        assert len(data["failures"]) == 2
        assert len(data["ranking_result"]["ranked_candidates"]) == 0


def test_rank_duplicate_candidate_ids() -> None:
    """Verify that duplicate candidate_ids in a request batch are processed and ordered stably."""

    def mock_orchestration_run(resume, jd, tools):
        return ScoreResult(
            resume_id="res",
            jd_id="jd-1",
            final_score=85,
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

    req_payload = {
        "raw_jd_text": "JD text",
        "resumes": [
            {"candidate_id": "A", "raw_resume_text": "resume skills"},
            {"candidate_id": "A", "raw_resume_text": "resume skills 2"},
        ],
    }

    with patch(
        "app.api.v1.endpoints.rank.run_orchestration",
        side_effect=mock_orchestration_run,
    ):
        response = client.post("/api/v1/rank", json=req_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_submitted"] == 2
        assert data["total_successful"] == 2
        candidates = data["ranking_result"]["ranked_candidates"]
        assert candidates[0]["candidate_id"] == "A"
        assert candidates[1]["candidate_id"] == "A"
        # They should be ranked 1 and 2
        assert candidates[0]["rank"] == 1
        assert candidates[1]["rank"] == 2


def test_rank_missing_resume_content_fails_row() -> None:
    """Verify that a row missing both parsed_resume and raw_resume_text records a failure for that row."""
    req_payload = {"raw_jd_text": "JD text", "resumes": [{"candidate_id": "empty_row"}]}
    response = client.post("/api/v1/rank", json=req_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_submitted"] == 1
    assert data["total_failed"] == 1
    assert data["failures"][0]["candidate_id"] == "empty_row"
    assert "No resume content provided" in data["failures"][0]["reason"]


def test_rank_zero_score_deterministic_order() -> None:
    """Verify tie-breaking on a final score of 0.0 is deterministic and alphabetical by candidate_id."""

    def mock_orchestration_run(resume, jd, tools):
        return ScoreResult(
            resume_id="res",
            jd_id="jd-1",
            final_score=0,
            feature_vector=FeatureVector(
                tfidf_score=0.0,
                embedding_score=0.0,
                skill_overlap_pct=0.0,
                exp_match=0.0,
                edu_match=0.0,
            ),
            scoring_confidence=0.5,
            confidence_level=ConfidenceLevel.LOW,
            parsing_confidence=0.5,
            pipeline_version="v3-hybrid",
        )

    req_payload = {
        "raw_jd_text": "JD text",
        "resumes": [
            {"candidate_id": "z", "raw_resume_text": "skills"},
            {"candidate_id": "a", "raw_resume_text": "skills"},
        ],
    }

    with patch(
        "app.api.v1.endpoints.rank.run_orchestration",
        side_effect=mock_orchestration_run,
    ):
        response = client.post("/api/v1/rank", json=req_payload)
        assert response.status_code == 200
        data = response.json()
        candidates = data["ranking_result"]["ranked_candidates"]
        assert candidates[0]["candidate_id"] == "a"
        assert candidates[1]["candidate_id"] == "z"

# ruff: noqa: E501
"""Unit and integration tests for Basic Auth and Recruiter Data Isolation (Phase 7.8)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.v1.endpoints.score import get_orchestrator_tools
import app.api.v1.endpoints.feedback as feedback_module
import app.api.v1.endpoints.metrics as metrics_module
from app.services.evaluation.ground_truth_schema import load_dataset
from app.schemas.scoring import ConfidenceLevel, FeatureVector, ScoreResult, SkillMatch, GapItem
from app.services.orchestration.agent_orchestrator import OrchestratorTools

client = TestClient(app, raise_server_exceptions=False)


# ============================ STUB INSTANCES FOR MOCKED SCORING ================
class StubExperienceMatcher:
    def match(self, *args, **kwargs) -> float:
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
            final_score=75.0,
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


_MOCK_TOOLS = OrchestratorTools(
    hybrid_scorer=StubHybridScorer(),  # type: ignore[arg-type]
    skill_matcher=StubSkillMatcher(),  # type: ignore[arg-type]
    taxonomy_entries=[],
    case_store=StubCaseStore(),  # type: ignore[arg-type]
    experience_matcher=StubExperienceMatcher(),  # type: ignore[arg-type]
)


# ============================ PATH & PERSISTENCE MOCK FIXTURE =================
@pytest.fixture(autouse=True)
def mock_persistence_paths(tmp_path: Path):
    """Isolate all dataset and outcome persistence paths for every test run."""
    orig_dataset_fb = feedback_module.DATASET_PATH
    orig_recruiter_fb = feedback_module.RECRUITER_PATH
    orig_audit_fb = feedback_module.AUDIT_LOG_PATH

    orig_dataset_mt = metrics_module.DATASET_PATH
    orig_history_mt = metrics_module.METRICS_HISTORY_PATH

    # Redirect to tmp path
    feedback_module.DATASET_PATH = tmp_path / "ground_truth_dataset.json"
    feedback_module.RECRUITER_PATH = tmp_path / "recruiter_feedback.json"
    feedback_module.AUDIT_LOG_PATH = tmp_path / "feedback_audit_log.jsonl"

    metrics_module.DATASET_PATH = tmp_path / "ground_truth_dataset.json"
    metrics_module.METRICS_HISTORY_PATH = tmp_path / "metrics_history.json"

    # Reset cache
    feedback_module._idempotency_keys.clear()

    yield

    # Restore originals
    feedback_module.DATASET_PATH = orig_dataset_fb
    feedback_module.RECRUITER_PATH = orig_recruiter_fb
    feedback_module.AUDIT_LOG_PATH = orig_audit_fb

    metrics_module.DATASET_PATH = orig_dataset_mt
    metrics_module.METRICS_HISTORY_PATH = orig_history_mt


@pytest.fixture(autouse=True)
def clean_overrides_and_mock_tools():
    """Ensure a clean dependency overrides state and mock expensive scoring tools."""
    app.dependency_overrides.clear()
    app.dependency_overrides[get_orchestrator_tools] = lambda: _MOCK_TOOLS
    yield
    app.dependency_overrides.clear()


# ============================ 1. Health Remains Public =======================
def test_health_remains_public() -> None:
    """Verify that /health remains unauthenticated and publicly reachable."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ============================ 2. Parse Remains Public ========================
def test_parse_remains_public() -> None:
    """Verify that /parse remains publicly accessible without credentials."""
    response = client.post(
        "/api/v1/parse",
        data={"jd_text": "JD text containing python development.", "document_type": "jd"},
    )
    assert response.status_code == 200
    assert "raw_text" in response.json()
    assert "python" in response.json()["raw_text"]


# ============================ 3. Score Remains Public ========================
def test_score_remains_public() -> None:
    """Verify that /score remains publicly accessible without credentials."""
    payload = {
        "raw_resume_text": "Jane Doe with python experience.",
        "raw_jd_text": "JD text",
    }
    response = client.post("/api/v1/score", json=payload)
    assert response.status_code == 200
    assert "score_result" in response.json()


# ============================ 4. Rank Unauthenticated Fails ==================
def test_rank_unauthenticated_fails() -> None:
    """Verify that /rank returns 401 when no auth headers are provided."""
    payload = {
        "raw_jd_text": "JD text",
        "resumes": [{"candidate_id": "c1", "raw_resume_text": "Jane Doe."}],
    }
    response = client.post("/api/v1/rank", json=payload)
    assert response.status_code == 401
    assert "basic" in response.headers.get("www-authenticate", "").lower()
    assert "authenticated" in response.json()["message"].lower()


# ============================ 5. Rank Invalid Auth Fails ====================
def test_rank_invalid_auth_fails() -> None:
    """Verify that /rank returns 401 on incorrect credentials."""
    payload = {
        "raw_jd_text": "JD text",
        "resumes": [{"candidate_id": "c1", "raw_resume_text": "Jane Doe."}],
    }
    response = client.post("/api/v1/rank", json=payload, auth=("recruiter_one", "wrong_password"))
    assert response.status_code == 401
    assert response.json()["code"] == "HTTP_401"
    assert "credentials" in response.json()["message"].lower()


# ============================ 6. Rank Valid Auth Success =====================
def test_rank_valid_auth_success() -> None:
    """Verify that /rank succeeds with valid basic credentials."""
    payload = {
        "raw_jd_text": "JD text",
        "resumes": [{"candidate_id": "c1", "raw_resume_text": "Jane Doe with python experience."}],
    }
    response = client.post("/api/v1/rank", json=payload, auth=("recruiter_one", "password123"))
    assert response.status_code == 200
    data = response.json()
    assert "ranking_result" in data
    assert data["total_submitted"] == 1


# ============================ 7. Feedback Unauthenticated Fails ==============
def test_feedback_unauthenticated_fails() -> None:
    """Verify that /feedback returns 401 when unauthenticated."""
    payload = {
        "feedback_type": "rater",
        "pair_id": "gt-01",
        "resume_id": "r1",
        "jd_id": "j1",
        "score": 85.0,
        "rater_id": "rater-A",
        "justification": "Valid justification text.",
    }
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 401


# ============================ 8. Feedback Invalid Auth Fails ================
def test_feedback_invalid_auth_fails() -> None:
    """Verify that /feedback returns 401 with invalid credentials."""
    payload = {
        "feedback_type": "rater",
        "pair_id": "gt-01",
        "resume_id": "r1",
        "jd_id": "j1",
        "score": 85.0,
        "rater_id": "rater-A",
        "justification": "Valid justification text.",
    }
    response = client.post("/api/v1/feedback", json=payload, auth=("invalid_user", "password"))
    assert response.status_code == 401


# ============================ 9. Feedback Rater Submission Success ==========
def test_feedback_rater_submission_success() -> None:
    """Verify rater feedback submission succeeds with valid basic credentials."""
    payload = {
        "feedback_type": "rater",
        "pair_id": "gt-01",
        "resume_id": "r1",
        "jd_id": "j1",
        "score": 85.0,
        "rater_id": "rater-A",
        "justification": "Valid justification text.",
    }
    response = client.post("/api/v1/feedback", json=payload, auth=("recruiter_one", "password123"))
    assert response.status_code == 200
    assert response.json()["status"] == "created"


# ============================ 10. Mismatched Recruiter ID Fails =============
def test_feedback_recruiter_mismatched_id_fails() -> None:
    """Verify Recruiter A cannot log feedback claiming to be Recruiter B."""
    payload = {
        "feedback_type": "recruiter",
        "score_id": "score-999",
        "actual_outcome": "interviewed",
        "recruiter_id": "recruiter_two",  # payload states recruiter_two
    }
    # Authenticate as recruiter_one
    response = client.post("/api/v1/feedback", json=payload, auth=("recruiter_one", "password123"))
    assert response.status_code == 403
    assert "Cannot log outcomes on behalf of another recruiter" in response.json()["message"]


# ============================ 11. Recruiter Feedback Success =================
def test_feedback_recruiter_submission_success() -> None:
    """Verify recruiter outcome logs successfully when credentials match payload."""
    payload = {
        "feedback_type": "recruiter",
        "score_id": "score-999",
        "actual_outcome": "interviewed",
        "recruiter_id": "recruiter_one",
    }
    response = client.post("/api/v1/feedback", json=payload, auth=("recruiter_one", "password123"))
    assert response.status_code == 200
    assert response.json()["status"] == "created"


# ============================ 12. Metrics Unauthenticated Fails =============
def test_metrics_unauthenticated_fails() -> None:
    """Verify that /metrics returns 401 when unauthenticated."""
    response = client.get("/api/v1/metrics")
    assert response.status_code == 401


# ============================ 13. Metrics Valid Auth Success ================
def test_metrics_valid_auth_success() -> None:
    """Verify that /metrics returns 200 when authenticated."""
    response = client.get("/api/v1/metrics", auth=("recruiter_one", "password123"))
    # Can return 200 or provisional/unready state
    assert response.status_code == 200
    assert "readiness_state" in response.json()


# ============================ 14. Adversarial Isolation Test =================
def test_adversarial_isolation_outcomes() -> None:
    """Verify that Recruiter B is forbidden from updating Recruiter A's stored outcome."""
    # 1. Recruiter A submits outcome for score-abc
    payload_a = {
        "feedback_type": "recruiter",
        "score_id": "score-abc",
        "actual_outcome": "interviewed",
        "recruiter_id": "recruiter_one",
    }
    res1 = client.post("/api/v1/feedback", json=payload_a, auth=("recruiter_one", "password123"))
    assert res1.status_code == 200
    assert res1.json()["status"] == "created"

    # 2. Recruiter B tries to overwrite or update the outcome for score-abc
    payload_b = {
        "feedback_type": "recruiter",
        "score_id": "score-abc",
        "actual_outcome": "hired",
        "recruiter_id": "recruiter_two",
    }
    res2 = client.post("/api/v1/feedback", json=payload_b, auth=("recruiter_two", "password456"))
    # Must fail with 403 Forbidden!
    assert res2.status_code == 403
    assert "Recruiter outcome ownership validation failed" in res2.json()["message"]

    # 3. Verify Recruiter A's outcome remains unchanged (interviewed, recruiter_one)
    with open(feedback_module.RECRUITER_PATH, "r", encoding="utf-8") as f:
        outcomes = json.load(f)
    assert len(outcomes) == 1
    assert outcomes[0]["score_id"] == "score-abc"
    assert outcomes[0]["recruiter_id"] == "recruiter_one"
    assert outcomes[0]["actual_outcome"] == "interviewed"


# ============================ 15. Shared Engine Integrity Test ================
def test_shared_engine_integrity() -> None:
    """Verify that the underlying scoring pipeline executes identically for all recruiters.

    This ensures that recruiter isolation does not partition the core AI engine.
    """
    payload = {
        "raw_jd_text": "JD text containing python and machine learning.",
        "resumes": [{"candidate_id": "c1", "raw_resume_text": "Jane Doe with python and ML experience."}],
    }

    # Recruiter A ranks
    res_a = client.post("/api/v1/rank", json=payload, auth=("recruiter_one", "password123"))
    assert res_a.status_code == 200
    score_a = res_a.json()["ranking_result"]["ranked_candidates"][0]["score_result"]["final_score"]

    # Recruiter B ranks identical candidate & JD
    res_b = client.post("/api/v1/rank", json=payload, auth=("recruiter_two", "password456"))
    assert res_b.status_code == 200
    score_b = res_b.json()["ranking_result"]["ranked_candidates"][0]["score_result"]["final_score"]

    # Must be character-for-character identical scores
    assert score_a == score_b

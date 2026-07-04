# ruff: noqa: E501
"""Unit and integration tests for the /feedback endpoint (Phase 7.5)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
import app.api.v1.endpoints.feedback as feedback_module
from app.services.evaluation.ground_truth_schema import load_dataset

client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def mock_feedback_paths(tmp_path: Path):
    """Automatically mock all persistence file paths for every test to isolate them.

    Clears the idempotency cache between tests as well.
    """
    orig_dataset = feedback_module.DATASET_PATH
    orig_recruiter = feedback_module.RECRUITER_PATH
    orig_audit = feedback_module.AUDIT_LOG_PATH

    feedback_module.DATASET_PATH = tmp_path / "ground_truth_dataset.json"
    feedback_module.RECRUITER_PATH = tmp_path / "recruiter_feedback.json"
    feedback_module.AUDIT_LOG_PATH = tmp_path / "feedback_audit_log.jsonl"

    # Reset in-memory cache
    feedback_module._idempotency_keys.clear()

    yield

    feedback_module.DATASET_PATH = orig_dataset
    feedback_module.RECRUITER_PATH = orig_recruiter
    feedback_module.AUDIT_LOG_PATH = orig_audit


@pytest.fixture(autouse=True)
def override_auth():
    """Use FastAPI dependency overrides to mock recruiter authentication."""
    from app.core.auth import get_current_recruiter, RecruiterAccount
    app.dependency_overrides[get_current_recruiter] = lambda: RecruiterAccount(
        account_id="company_a", recruiter_id="recruiter-12"
    )
    yield
    app.dependency_overrides.pop(get_current_recruiter, None)


# 1. Successful Rater Submission
def test_rater_feedback_success() -> None:
    payload = {
        "feedback_type": "rater",
        "pair_id": "gt-01",
        "resume_id": "res-101",
        "jd_id": "jd-202",
        "score": 85.5,
        "rater_id": "rater-A",
        "justification": "Candidate has excellent experience with python and docker.",
        "case_type": "clear_fit",
    }
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["stored_id"] == "gt-01"
    assert data["feedback_type"] == "rater"
    assert data["status"] == "created"
    assert data["details"]["score"] == 85.5
    assert data["progress"]["total_target"] == 1  # only 1 pair exists now

    # Verify persistence
    assert feedback_module.DATASET_PATH.exists()
    dataset = load_dataset(str(feedback_module.DATASET_PATH))
    assert len(dataset.pairs) == 1
    assert dataset.pairs[0].pair_id == "gt-01"
    assert dataset.pairs[0].rater_scores[0].score == 85.5
    assert dataset.pairs[0].reconciled_score == 85.5


# 2. Progress Snapshot Check
def test_rater_feedback_progress_snapshot() -> None:
    # Submit first rating
    client.post(
        "/api/v1/feedback",
        json={
            "feedback_type": "rater",
            "pair_id": "gt-01",
            "resume_id": "r1",
            "jd_id": "j1",
            "score": 70.0,
            "rater_id": "rater-A",
            "justification": "Valid basic requirements matched.",
        },
    )
    # Submit second rating for same pair from different rater
    response = client.post(
        "/api/v1/feedback",
        json={
            "feedback_type": "rater",
            "pair_id": "gt-01",
            "resume_id": "r1",
            "jd_id": "j1",
            "score": 75.0,
            "rater_id": "rater-B",
            "justification": "Valid basic requirements matched.",
        },
    )
    data = response.json()
    # Still not fully covered (< 3 raters)
    assert data["progress"]["pairs_with_full_rater_coverage"] == 0
    assert data["progress"]["pairs_still_needing_raters"] == 1

    # Submit third rater rating to complete coverage
    response = client.post(
        "/api/v1/feedback",
        json={
            "feedback_type": "rater",
            "pair_id": "gt-01",
            "resume_id": "r1",
            "jd_id": "j1",
            "score": 80.0,
            "rater_id": "rater-C",
            "justification": "Valid basic requirements matched.",
        },
    )
    data = response.json()
    assert data["progress"]["pairs_with_full_rater_coverage"] == 1
    assert data["progress"]["pairs_still_needing_raters"] == 0


# 3. Successful Recruiter Outcome Submission
def test_recruiter_feedback_success() -> None:
    payload = {
        "feedback_type": "recruiter",
        "score_id": "score-999",
        "actual_outcome": "hired",
        "recruiter_id": "recruiter-12",
    }
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["stored_id"] == "score-999"
    assert data["feedback_type"] == "recruiter"
    assert data["status"] == "created"
    assert data["details"]["actual_outcome"] == "hired"

    # Verify persistence
    assert feedback_module.RECRUITER_PATH.exists()
    with open(feedback_module.RECRUITER_PATH, "r", encoding="utf-8") as f:
        outcomes = json.load(f)
    assert len(outcomes) == 1
    assert outcomes[0]["score_id"] == "score-999"
    assert outcomes[0]["actual_outcome"] == "hired"


# 4. Out of Range Rater Score Low
def test_rater_feedback_out_of_range_low() -> None:
    payload = {
        "feedback_type": "rater",
        "pair_id": "gt-01",
        "resume_id": "r1",
        "jd_id": "j1",
        "score": -5.0,
        "rater_id": "rater-A",
        "justification": "Invalid score underflow.",
    }
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 422


# 5. Out of Range Rater Score High
def test_rater_feedback_out_of_range_high() -> None:
    payload = {
        "feedback_type": "rater",
        "pair_id": "gt-01",
        "resume_id": "r1",
        "jd_id": "j1",
        "score": 101.5,
        "rater_id": "rater-A",
        "justification": "Invalid score overflow.",
    }
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 422


# 6. Missing Justification
def test_rater_feedback_missing_justification() -> None:
    payload = {
        "feedback_type": "rater",
        "pair_id": "gt-01",
        "resume_id": "r1",
        "jd_id": "j1",
        "score": 85.0,
        "rater_id": "rater-A",
        "justification": "   ",
    }
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 422


# 7. Justification Cannot Be Numeric
def test_rater_feedback_numeric_justification() -> None:
    payload = {
        "feedback_type": "rater",
        "pair_id": "gt-01",
        "resume_id": "r1",
        "jd_id": "j1",
        "score": 85.0,
        "rater_id": "rater-A",
        "justification": "85.0",
    }
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 422
    assert "cannot be a numeric value" in response.text


# 8. Unrecognized Recruiter Outcome
def test_recruiter_feedback_invalid_outcome() -> None:
    payload = {
        "feedback_type": "recruiter",
        "score_id": "score-999",
        "actual_outcome": "maybe",
        "recruiter_id": "recruiter-12",
    }
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 422


# 9. Header-based Idempotency Key
def test_rater_feedback_idempotency_key_header() -> None:
    payload = {
        "feedback_type": "rater",
        "pair_id": "gt-01",
        "resume_id": "r1",
        "jd_id": "j1",
        "score": 85.0,
        "rater_id": "rater-A",
        "justification": "Valid reasoning provided.",
    }
    headers = {"X-Idempotency-Key": "unique-idemp-key-1"}

    # First request
    r1 = client.post("/api/v1/feedback", json=payload, headers=headers)
    assert r1.status_code == 200
    assert r1.json()["status"] == "created"

    # Second request with same header
    r2 = client.post("/api/v1/feedback", json=payload, headers=headers)
    assert r2.status_code == 200
    assert r2.json()["status"] == "created"  # Cached copy returned

    # Verify only one score actually exists in file
    dataset = load_dataset(str(feedback_module.DATASET_PATH))
    assert len(dataset.pairs[0].rater_scores) == 1


# 10. Natural Key Idempotency (duplicate)
def test_rater_feedback_natural_key_idempotency() -> None:
    payload = {
        "feedback_type": "rater",
        "pair_id": "gt-01",
        "resume_id": "r1",
        "jd_id": "j1",
        "score": 85.0,
        "rater_id": "rater-A",
        "justification": "Valid reasoning provided.",
    }

    # First submission
    r1 = client.post("/api/v1/feedback", json=payload)
    assert r1.json()["status"] == "created"

    # Second identical submission (no idempotency header, relies on natural key check)
    r2 = client.post("/api/v1/feedback", json=payload)
    assert r2.status_code == 200
    assert r2.json()["status"] == "duplicate"

    # Verify no double counting
    dataset = load_dataset(str(feedback_module.DATASET_PATH))
    assert len(dataset.pairs[0].rater_scores) == 1


# 11. Correction Updates Rater Score
def test_rater_feedback_correction_updates() -> None:
    payload = {
        "feedback_type": "rater",
        "pair_id": "gt-01",
        "resume_id": "r1",
        "jd_id": "j1",
        "score": 80.0,
        "rater_id": "rater-A",
        "justification": "Initial rating reasoning.",
    }
    client.post("/api/v1/feedback", json=payload)

    # Submitting updated score/justification for same rater/pair
    correction_payload = payload.copy()
    correction_payload["score"] = 90.0
    correction_payload["justification"] = "Corrected rating after closer inspection."

    response = client.post("/api/v1/feedback", json=correction_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "updated"
    assert data["details"]["score"] == 90.0

    # Verify update in persistence
    dataset = load_dataset(str(feedback_module.DATASET_PATH))
    assert len(dataset.pairs[0].rater_scores) == 1
    assert dataset.pairs[0].rater_scores[0].score == 90.0
    assert dataset.pairs[0].reconciled_score == 90.0


# 12. Recruiter Natural Key Idempotency
def test_recruiter_feedback_natural_key_idempotency() -> None:
    payload = {
        "feedback_type": "recruiter",
        "score_id": "score-999",
        "actual_outcome": "rejected",
        "recruiter_id": "recruiter-12",
    }
    client.post("/api/v1/feedback", json=payload)
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "duplicate"

    with open(feedback_module.RECRUITER_PATH, "r", encoding="utf-8") as f:
        outcomes = json.load(f)
    assert len(outcomes) == 1


# 13. Recruiter Outcome Correction
def test_recruiter_feedback_correction() -> None:
    payload = {
        "feedback_type": "recruiter",
        "score_id": "score-999",
        "actual_outcome": "interviewed",
        "recruiter_id": "recruiter-12",
    }
    client.post("/api/v1/feedback", json=payload)

    # Update outcome
    payload["actual_outcome"] = "hired"
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "updated"

    with open(feedback_module.RECRUITER_PATH, "r", encoding="utf-8") as f:
        outcomes = json.load(f)
    assert len(outcomes) == 1
    assert outcomes[0]["actual_outcome"] == "hired"


# 14. Complete Provenance inside Audit Log
def test_provenance_logged_in_audit_log() -> None:
    payload = {
        "feedback_type": "rater",
        "pair_id": "gt-01",
        "resume_id": "r1",
        "jd_id": "j1",
        "score": 85.0,
        "rater_id": "rater-A",
        "justification": "High quality matching.",
    }
    client.post("/api/v1/feedback", json=payload)

    assert feedback_module.AUDIT_LOG_PATH.exists()
    lines = feedback_module.AUDIT_LOG_PATH.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    log_entry = json.loads(lines[0])

    assert "id" in log_entry
    assert log_entry["feedback_type"] == "rater"
    assert log_entry["submitter_identity"] == "rater-A"
    assert log_entry["role"] == "rater"
    assert "timestamp" in log_entry
    assert log_entry["raw_payload"] == payload
    assert log_entry["normalized_payload"]["score"] == 85.0


# 15. Separate Storage Paths (No cross-contamination)
def test_separate_storage_paths_no_cross_contamination() -> None:
    rater_payload = {
        "feedback_type": "rater",
        "pair_id": "gt-01",
        "resume_id": "r1",
        "jd_id": "j1",
        "score": 85.0,
        "rater_id": "rater-A",
        "justification": "Excellent profile.",
    }
    recruiter_payload = {
        "feedback_type": "recruiter",
        "score_id": "score-999",
        "actual_outcome": "hired",
        "recruiter_id": "recruiter-12",
    }

    client.post("/api/v1/feedback", json=rater_payload)
    client.post("/api/v1/feedback", json=recruiter_payload)

    # Dataset file must only contain the rater payload
    dataset = load_dataset(str(feedback_module.DATASET_PATH))
    assert len(dataset.pairs) == 1
    assert dataset.pairs[0].pair_id == "gt-01"

    # Recruiter file must only contain the recruiter outcome
    with open(feedback_module.RECRUITER_PATH, "r", encoding="utf-8") as f:
        outcomes = json.load(f)
    assert len(outcomes) == 1
    assert outcomes[0]["score_id"] == "score-999"


# 16. Multi-Submission Collection Progress Sequence
def test_live_collection_progress_computation() -> None:
    # Check progress sequence against multiple submissions
    for i in range(5):
        # Pairs: gt-0, gt-1, gt-2, gt-3, gt-4
        # We submit 3 ratings for each of the first 3 pairs (gt-0 to gt-2)
        # We submit 1 rating for the remaining 2 pairs (gt-3 to gt-4)
        for rater in ["rater-X", "rater-Y", "rater-Z"]:
            client.post(
                "/api/v1/feedback",
                json={
                    "feedback_type": "rater",
                    "pair_id": f"gt-{i}",
                    "resume_id": f"r-{i}",
                    "jd_id": f"j-{i}",
                    "score": 80.0,
                    "rater_id": rater,
                    "justification": "Consistent mock rating.",
                },
            )
            # Break early for gt-3 and gt-4 to leave them partially rated
            if i >= 3:
                break

    # Re-fetch dataset to verify progress computation manually
    dataset = load_dataset(str(feedback_module.DATASET_PATH))
    progress = feedback_module.check_ground_truth_collection_progress(dataset)

    # pairs gt-0, gt-1, gt-2 have 3 ratings
    assert progress["pairs_with_full_rater_coverage"] == 3
    # pairs gt-3, gt-4 have 1 rating each (<3)
    assert progress["pairs_still_needing_raters"] == 2
    assert progress["total_target"] == 5

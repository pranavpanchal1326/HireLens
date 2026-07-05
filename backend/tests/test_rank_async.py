# ruff: noqa: E501
"""Tests for the Phase R5 async batch-ranking delivery (job submission + polling)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.v1.endpoints.score import get_orchestrator_tools
from app.api.v1.endpoints.rank import get_job_store
from app.services.ranking.job_store import RankJobStore

# Reuse the mocked scoring tools so tests never load real ML models.
from tests.test_auth_isolation import _MOCK_TOOLS

AUTH_A = ("recruiter_one", "password123")   # account: company_a
AUTH_B = ("recruiter_two", "password456")   # account: company_b


@pytest.fixture
def client(tmp_path):
    store = RankJobStore(tmp_path / "rank_jobs.json")
    app.dependency_overrides[get_orchestrator_tools] = lambda: _MOCK_TOOLS
    app.dependency_overrides[get_job_store] = lambda: store
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def _batch(n: int, bad_index: int | None = None) -> dict:
    resumes = []
    for i in range(n):
        text = "" if i == bad_index else "Jane with python and machine learning experience."
        resumes.append({"candidate_id": f"c{i}", "raw_resume_text": text})
    return {"raw_jd_text": "python and machine learning role", "resumes": resumes}


def test_small_batch_stays_synchronous(client) -> None:
    """<=50 resumes → immediate RankResponse, no job envelope (unchanged contract)."""
    resp = client.post("/api/v1/rank", json=_batch(3), auth=AUTH_A)
    assert resp.status_code == 200
    body = resp.json()
    assert "ranking_result" in body and "job_id" not in body


def test_large_batch_returns_job_envelope(client) -> None:
    """>50 resumes → 202 with job_id + estimated_items (non-blocking submission)."""
    resp = client.post("/api/v1/rank", json=_batch(60), auth=AUTH_A)
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert body["estimated_items"] == 60
    assert "job_id" in body


def test_async_result_matches_sync_schema(client) -> None:
    """Polling a completed job returns the EXACT RankResponse shape."""
    job_id = client.post("/api/v1/rank", json=_batch(55), auth=AUTH_A).json()["job_id"]
    status = client.get(f"/api/v1/rank/{job_id}/status", auth=AUTH_A).json()
    assert status["status"] == "complete"
    assert status["completed"] == 55 and status["total"] == 55
    result = status["result"]
    # Same client-facing contract as the synchronous /rank response.
    for key in ("ranking_result", "pipeline_maturity", "total_submitted", "total_successful", "total_failed", "failures"):
        assert key in result
    assert result["total_submitted"] == 55


def test_partial_failure_does_not_fail_whole_batch(client) -> None:
    """One empty resume among 55 → that one fails, the rest still rank."""
    job_id = client.post("/api/v1/rank", json=_batch(55, bad_index=7), auth=AUTH_A).json()["job_id"]
    result = client.get(f"/api/v1/rank/{job_id}/status", auth=AUTH_A).json()["result"]
    assert result["total_failed"] == 1
    assert result["total_successful"] == 54
    assert any(f["candidate_id"] == "c7" for f in result["failures"])


def test_status_endpoint_requires_auth(client) -> None:
    job_id = client.post("/api/v1/rank", json=_batch(55), auth=AUTH_A).json()["job_id"]
    resp = client.get(f"/api/v1/rank/{job_id}/status")  # no auth
    assert resp.status_code == 401


def test_idor_other_company_cannot_read_job(client) -> None:
    """Company B must not be able to read Company A's job (404, no existence leak)."""
    job_id = client.post("/api/v1/rank", json=_batch(55), auth=AUTH_A).json()["job_id"]
    resp = client.get(f"/api/v1/rank/{job_id}/status", auth=AUTH_B)
    assert resp.status_code == 404


def test_unknown_job_is_404(client) -> None:
    resp = client.get("/api/v1/rank/nonexistent-id/status", auth=AUTH_A)
    assert resp.status_code == 404

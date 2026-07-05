# ruff: noqa: E501
"""Tests for the Phase R3 freemium rate-limiter."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

import app.api.v1.endpoints.score as score_module
from app.main import app
from app.api.v1.endpoints.score import get_orchestrator_tools
from app.services.ratelimit.limiter import FreemiumRateLimiter, RateLimitResult
from app.services.ratelimit.scan_store import InMemoryScanStore, JSONFileScanStore

# Reuse the mocked scoring tools so endpoint tests never load real ML models.
from tests.test_auth_isolation import _MOCK_TOOLS

T0 = datetime(2026, 1, 1, tzinfo=UTC)


# ============================ PURE LOGIC ====================================
def test_under_limit_allows_and_decrements() -> None:
    limiter = FreemiumRateLimiter(InMemoryScanStore(), limit=3, window_days=30)
    r1 = limiter.check_and_increment("u1", now=T0)
    r2 = limiter.check_and_increment("u1", now=T0)
    r3 = limiter.check_and_increment("u1", now=T0)
    assert [r.allowed for r in (r1, r2, r3)] == [True, True, True]
    assert [r.remaining for r in (r1, r2, r3)] == [2, 1, 0]


def test_at_limit_denies_with_reset_date() -> None:
    limiter = FreemiumRateLimiter(InMemoryScanStore(), limit=3, window_days=30)
    for _ in range(3):
        limiter.check_and_increment("u1", now=T0)
    denied = limiter.check_and_increment("u1", now=T0)
    assert denied.allowed is False
    assert denied.remaining == 0
    assert denied.resets_at == T0 + timedelta(days=30)


def test_identifiers_tracked_independently() -> None:
    limiter = FreemiumRateLimiter(InMemoryScanStore(), limit=1, window_days=30)
    a = limiter.check_and_increment("A", now=T0)
    b = limiter.check_and_increment("B", now=T0)
    assert a.allowed and b.allowed  # B is not affected by A exhausting its quota
    assert limiter.check_and_increment("A", now=T0).allowed is False
    assert limiter.check_and_increment("B", now=T0).allowed is False


def test_window_resets_after_expiry() -> None:
    limiter = FreemiumRateLimiter(InMemoryScanStore(), limit=1, window_days=30)
    assert limiter.check_and_increment("u1", now=T0).allowed is True
    assert limiter.check_and_increment("u1", now=T0).allowed is False
    # One day past the window → fresh quota.
    later = T0 + timedelta(days=31)
    fresh = limiter.check_and_increment("u1", now=later)
    assert fresh.allowed is True
    assert fresh.remaining == 0
    assert fresh.resets_at == later + timedelta(days=30)


def test_denied_call_does_not_consume() -> None:
    limiter = FreemiumRateLimiter(InMemoryScanStore(), limit=1, window_days=30)
    limiter.check_and_increment("u1", now=T0)
    first_denied = limiter.check_and_increment("u1", now=T0)
    second_denied = limiter.check_and_increment("u1", now=T0)
    # Reset date is stable across repeated denied calls (no window churn).
    assert first_denied.resets_at == second_denied.resets_at


# ============================ INTERIM STORAGE FLAG ==========================
def test_file_store_survives_restart(tmp_path) -> None:
    """The interim file store persists across process restarts (unlike in-memory).

    KNOWN GAP (documented, not 'done'): the file store is NOT safe under
    concurrent writes from multiple workers — a counter update can be lost under a
    race. Adequate for a single-worker free-tier deployment only; swap to Redis/DB
    before scaling out. See scan_store.py.
    """
    path = tmp_path / "scans.json"
    limiter1 = FreemiumRateLimiter(JSONFileScanStore(path), limit=3, window_days=30)
    limiter1.check_and_increment("u1", now=T0)
    limiter1.check_and_increment("u1", now=T0)
    # Simulate a restart: brand-new limiter + store instance, same file.
    limiter2 = FreemiumRateLimiter(JSONFileScanStore(path), limit=3, window_days=30)
    r = limiter2.check_and_increment("u1", now=T0)
    assert r.remaining == 0  # count carried over: 2 prior + this = 3


# ============================ ENDPOINT WIRING ===============================
@pytest.fixture
def client():
    original = score_module._RATE_LIMITER
    app.dependency_overrides[get_orchestrator_tools] = lambda: _MOCK_TOOLS
    yield TestClient(app, raise_server_exceptions=False)
    score_module._RATE_LIMITER = original
    app.dependency_overrides.clear()


def test_score_returns_429_when_limit_reached(client) -> None:
    # Force a limit of 1 and pre-consume it for this anon id.
    limiter = FreemiumRateLimiter(InMemoryScanStore(), limit=1, window_days=30)
    limiter.check_and_increment("anon:device-xyz")  # exhaust the single free scan
    score_module._RATE_LIMITER = limiter

    resp = client.post(
        "/api/v1/score",
        json={"raw_resume_text": "Jane with python", "raw_jd_text": "python role"},
        headers={"X-Anon-Id": "device-xyz"},
    )
    assert resp.status_code == 429
    body = resp.json()
    assert body["reason"] == "FREEMIUM_LIMIT_REACHED"
    assert body["remaining"] == 0
    assert "resets_at" in body


def test_recruiter_rank_unaffected_by_scan_limit(client) -> None:
    """/rank stays available regardless of anonymous scan exhaustion."""
    # Exhaust the seeker scan quota entirely.
    limiter = FreemiumRateLimiter(InMemoryScanStore(), limit=1, window_days=30)
    limiter.check_and_increment("anon:someone")
    score_module._RATE_LIMITER = limiter

    payload = {
        "raw_jd_text": "python role",
        "resumes": [{"candidate_id": "c1", "raw_resume_text": "Jane with python."}],
    }
    # Multiple recruiter calls, all must succeed — never 429.
    for _ in range(5):
        resp = client.post(
            "/api/v1/rank", json=payload, auth=("recruiter_one", "password123")
        )
        assert resp.status_code == 200
        assert resp.status_code != 429


def test_result_type_shape() -> None:
    limiter = FreemiumRateLimiter(InMemoryScanStore(), limit=3)
    result = limiter.check_and_increment("u1", now=T0)
    assert isinstance(result, RateLimitResult)
    assert isinstance(result.resets_at, datetime)

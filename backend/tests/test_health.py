"""Smoke test for the /health liveness route."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == settings.APP_NAME
    assert data["version"] == "0.1.0"
    assert "timestamp" in data

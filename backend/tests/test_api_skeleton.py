# ruff: noqa: E501
"""Unit tests for the FastAPI skeleton, error handlers, and factory (Phase 7.1)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.core.config import settings
from app.main import create_app


class DummyModel(BaseModel):
    name: str
    age: int


def test_app_factory_repeated() -> None:
    """Verify that the app factory produces independent, configured instances."""
    app1 = create_app()
    app2 = create_app()
    assert app1 is not app2
    assert app1.title == settings.APP_NAME
    assert app2.title == settings.APP_NAME
    assert app1.version == "0.1.0"
    assert app2.version == "0.1.0"


def test_openapi_metadata() -> None:
    """Verify that OpenAPI documentation metadata matches the project settings."""
    app = create_app()
    assert app.title == settings.APP_NAME
    assert "Orchestrated Resume Evaluation" in app.description
    assert app.version == "0.1.0"


def test_error_handlers() -> None:
    """Verify global error handlers format responses in the standardized JSON envelope."""
    app = create_app()
    router = APIRouter()

    @router.get("/test-http-error")
    def http_error() -> None:
        raise HTTPException(status_code=400, detail="Custom HTTP 400 error details")

    @router.get("/test-unhandled-error")
    def unhandled_error() -> None:
        raise ValueError("Simulated unexpected crash")

    @router.post("/test-validation-error")
    def validation_error(body: DummyModel) -> dict[str, bool]:
        return {"ok": True}

    app.include_router(router)
    client = TestClient(app, raise_server_exceptions=False)

    # 1. Test HTTP Exception Handler (HTTP_400)
    resp_http = client.get("/test-http-error")
    assert resp_http.status_code == 400
    data_http = resp_http.json()
    assert data_http["code"] == "HTTP_400"
    assert data_http["message"] == "Custom HTTP 400 error details"
    assert "request_id" in data_http

    # 2. Test Request Validation Error Handler (VALIDATION_ERROR)
    resp_val = client.post(
        "/test-validation-error", json={"name": "John", "age": "not-an-int"}
    )
    assert resp_val.status_code == 422
    data_val = resp_val.json()
    assert data_val["code"] == "VALIDATION_ERROR"
    assert "details" in data_val
    assert "request_id" in data_val

    # 3. Test General/Unhandled Exception Handler (INTERNAL_SERVER_ERROR)
    resp_gen = client.get("/test-unhandled-error")
    assert resp_gen.status_code == 500
    data_gen = resp_gen.json()
    assert data_gen["code"] == "INTERNAL_SERVER_ERROR"
    assert "unhandled internal server error" in data_gen["message"]
    assert "request_id" in data_gen

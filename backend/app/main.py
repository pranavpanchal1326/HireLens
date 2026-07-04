"""HireLens FastAPI application entrypoint.

Phase 0.1 scope: app instance, CORS, versioned router wiring, and a single
``/health`` route. No business logic lives here.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging_config import configure_logging

configure_logging()

app = FastAPI(title=settings.APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe used by tests, Docker, and future deployment platforms."""
    return {"status": "ok", "app": settings.APP_NAME}

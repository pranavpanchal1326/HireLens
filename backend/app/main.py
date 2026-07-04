"""HireLens FastAPI application entrypoint.

Phase 7.1 scope: app instance factory, CORS, global error handling,
and structured health probe.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging_config import configure_logging

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    """Register global exception handlers for standardized error responses."""

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "request_id": request.headers.get("x-request-id"),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        return JSONResponse(
            status_code=422,
            content={
                "code": "VALIDATION_ERROR",
                "message": "Input validation failed.",
                "details": exc.errors(),
                "request_id": request.headers.get("x-request-id"),
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error occurred: %s", str(exc))
        return JSONResponse(
            status_code=500,
            content={
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unhandled internal server error occurred.",
                "request_id": request.headers.get("x-request-id"),
            },
        )


def create_app() -> FastAPI:
    """Application factory for configuring and returning the FastAPI app instance."""
    configure_logging()

    app = FastAPI(
        title=settings.APP_NAME,
        description="HireLens API - Recruiter Orchestrated Resume Evaluation Pipeline",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers (Aggregate router api_router)
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    # Add custom global error handlers
    register_error_handlers(app)

    # Health check route
    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe. Extensible to check databases/caches in future phases."""
        return {
            "status": "ok",
            "app": settings.APP_NAME,
            "version": "0.1.0",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    return app


# Instantiate app at the module level for Uvicorn server and test discovery
app = create_app()

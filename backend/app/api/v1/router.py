# ruff: noqa: E501
"""Aggregate router for all v1 API endpoints.

Currently empty by design (Phase 0.1). Future phases attach their sub-routers
here via ``api_router.include_router(...)`` so ``main.py`` only ever wires in
this single aggregator.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints.feedback import router as feedback_router
from app.api.v1.endpoints.metrics import router as metrics_router
from app.api.v1.endpoints.parse import router as parse_router
from app.api.v1.endpoints.rank import router as rank_router
from app.api.v1.endpoints.score import router as score_router

api_router = APIRouter()

api_router.include_router(parse_router, tags=["parsing"])
api_router.include_router(score_router, tags=["scoring"])
api_router.include_router(rank_router, tags=["ranking"])
api_router.include_router(feedback_router, prefix="/feedback", tags=["feedback"])
api_router.include_router(metrics_router, prefix="/metrics", tags=["metrics"])


# Aggregate router for all v1 API endpoints.
# Future endpoints will be attached here:
#   - api_router.include_router(metrics_router, prefix="/metrics", tags=["metrics"]) (Phase 7.6)


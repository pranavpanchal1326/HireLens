# ruff: noqa: E501
"""Aggregate router for all v1 API endpoints.

Wires all Phase 7 sub-routers (/parse, /score, /rank, /feedback, /metrics)
into the single aggregator that ``main.py`` mounts under ``/api/v1``.
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

# ruff: noqa: E501
"""Aggregate router for all v1 API endpoints.

Currently empty by design (Phase 0.1). Future phases attach their sub-routers
here via ``api_router.include_router(...)`` so ``main.py`` only ever wires in
this single aggregator.
"""

from __future__ import annotations

from fastapi import APIRouter

api_router = APIRouter()

# Aggregate router for all v1 API endpoints.
# Future endpoints will be attached here:
#   - api_router.include_router(parse_router, prefix="/parse", tags=["parsing"]) (Phase 7.2)
#   - api_router.include_router(score_router, prefix="/score", tags=["scoring"]) (Phase 7.3)
#   - api_router.include_router(rank_router, prefix="/rank", tags=["ranking"]) (Phase 7.4)
#   - api_router.include_router(feedback_router, prefix="/feedback", tags=["feedback"]) (Phase 7.5)
#   - api_router.include_router(metrics_router, prefix="/metrics", tags=["metrics"]) (Phase 7.6)

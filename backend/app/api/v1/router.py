"""Aggregate router for all v1 API endpoints.

Currently empty by design (Phase 0.1). Future phases attach their sub-routers
here via ``api_router.include_router(...)`` so ``main.py`` only ever wires in
this single aggregator.
"""

from __future__ import annotations

from fastapi import APIRouter

api_router = APIRouter()

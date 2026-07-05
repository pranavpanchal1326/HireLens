"""Shared pytest fixtures.

The freemium rate limiter (Phase R3) gates the anonymous /score path. Pre-existing
suites call /score many times and must not trip it, so by default we swap the
limiter singleton for a permissive in-memory one. This patches the module-level
singleton directly (not via dependency_overrides) so it survives the
``dependency_overrides.clear()`` that other suites' fixtures perform. Tests that
actually exercise the limit set their own limiter explicitly.
"""

from __future__ import annotations

import pytest

import app.api.v1.endpoints.score as score_module
from app.services.ratelimit.limiter import FreemiumRateLimiter
from app.services.ratelimit.scan_store import InMemoryScanStore


@pytest.fixture(autouse=True)
def neutralize_rate_limiter():
    """Default: an effectively-unlimited limiter so unrelated tests aren't gated."""
    original = score_module._RATE_LIMITER
    score_module._RATE_LIMITER = FreemiumRateLimiter(
        InMemoryScanStore(), limit=10_000
    )
    yield
    score_module._RATE_LIMITER = original

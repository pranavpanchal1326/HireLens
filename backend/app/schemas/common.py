"""Shared/common data contracts.

Maps to: Design Blueprint §10.10 — error messages surfaced to the frontend must
be warm and blameless, never a raw stack trace.

Produced by: all API routes (error paths). Consumed by: Phase 8 (frontend).
"""

from __future__ import annotations

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Canonical error envelope returned to clients.

    Maps to: Design Blueprint §10.10 — ``message`` must always be blameless and
    human (e.g. "we_couldnt_read_this_resume_clearly"), never an exposed stack
    trace. ``details`` may carry structured, non-sensitive diagnostics.
    """

    error_code: str
    message: str
    details: dict[str, object] | None = None

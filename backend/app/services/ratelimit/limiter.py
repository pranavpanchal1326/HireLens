"""Freemium scan-limit logic (Phase R3).

PRD §3.1: "Freemium model: 3 free scans/month." This module owns the counting
and window math ONLY. Identification (which anon_id) and transport (the /score
route, the 429) live at the endpoint layer; storage is injected. Keeping this
pure makes it unit-testable with a fixed clock and an in-memory store, and lets
the interim file store be swapped later without touching this math.

WINDOW INTERPRETATION (flagged, not silently chosen): "per month" is implemented
as a ROLLING 30-day window measured from each identifier's FIRST scan — more
abuse-resistant than a calendar-month reset (which would let a user burn 3 scans
on the 31st and 3 more on the 1st). Change WINDOW_DAYS / this policy if the PRD
is later clarified to mean calendar month.

What this module does NOT do: it does not identify users, does not read headers,
does not fingerprint, and does not touch scoring/parsing/RAG/model or recruiter
auth. It only answers "has THIS identifier used its free scans yet?"
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.services.ratelimit.scan_store import ScanStore

DEFAULT_SCAN_LIMIT = 3
DEFAULT_WINDOW_DAYS = 30

_COUNT = "count"
_WINDOW_START = "window_start"


@dataclass(frozen=True)
class RateLimitResult:
    """Outcome of a scan-limit check."""

    allowed: bool
    remaining: int
    resets_at: datetime


class FreemiumRateLimiter:
    """Counts anonymous scans per identifier over a rolling window."""

    def __init__(
        self,
        store: ScanStore,
        limit: int = DEFAULT_SCAN_LIMIT,
        window_days: int = DEFAULT_WINDOW_DAYS,
    ) -> None:
        self._store = store
        self._limit = limit
        self._window = timedelta(days=window_days)

    def _parse_start(self, record: dict) -> datetime:
        return datetime.fromisoformat(record[_WINDOW_START])

    def check_and_increment(
        self, anon_id: str, now: datetime | None = None
    ) -> RateLimitResult:
        """Consume one scan for ``anon_id`` if allowed; otherwise report denial.

        On ALLOWED the counter is incremented and persisted. On DENIED nothing is
        written. ``now`` is injectable so tests can drive the rolling window
        deterministically.
        """
        now = now or datetime.now(UTC)
        record = self._store.get(anon_id)

        # New identifier, or the previous window has fully elapsed → fresh window.
        if record is None or now >= self._parse_start(record) + self._window:
            new_record = {_COUNT: 1, _WINDOW_START: now.isoformat()}
            self._store.set(anon_id, new_record)
            return RateLimitResult(
                allowed=True,
                remaining=self._limit - 1,
                resets_at=now + self._window,
            )

        window_start = self._parse_start(record)
        resets_at = window_start + self._window
        used = int(record[_COUNT])

        if used >= self._limit:
            # Over the limit — do NOT increment; report when it frees up.
            return RateLimitResult(allowed=False, remaining=0, resets_at=resets_at)

        record[_COUNT] = used + 1
        self._store.set(anon_id, record)
        return RateLimitResult(
            allowed=True,
            remaining=self._limit - record[_COUNT],
            resets_at=resets_at,
        )

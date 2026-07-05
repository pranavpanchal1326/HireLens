"""Storage backends for the freemium scan counter (Phase R3).

The limiter logic (``limiter.py``) is pure and depends only on the ``ScanStore``
Protocol below, so the interim file store here can be swapped for a real DB /
Redis later WITHOUT changing the limit math.

INTERIM-STORAGE FLAG (loud, per R3 constraints): ``JSONFileScanStore`` is a
file-backed JSON dict. It DOES survive a server restart (unlike pure in-memory),
but it is NOT safe under concurrent writes from multiple processes/workers — the
last writer wins and a counter update can be lost under a race. It is adequate
for a single-worker POC/free-tier deployment ONLY. This is a KNOWN GAP, tracked
as such, not a production-grade limiter. Swap to an atomic store (Redis INCR, or
a DB row with a transaction) before running multiple workers.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Protocol


class ScanStore(Protocol):
    """Minimal persistence contract the limiter depends on."""

    def get(self, anon_id: str) -> dict | None:
        """Return the stored record for ``anon_id`` or None."""
        ...

    def set(self, anon_id: str, record: dict) -> None:
        """Persist the record for ``anon_id``."""
        ...


class InMemoryScanStore:
    """Non-persistent store. For tests and truly-ephemeral dev only.

    Data is lost on restart — never use as the real limiter backing store.
    """

    def __init__(self) -> None:
        self._data: dict[str, dict] = {}

    def get(self, anon_id: str) -> dict | None:
        rec = self._data.get(anon_id)
        return dict(rec) if rec is not None else None

    def set(self, anon_id: str, record: dict) -> None:
        self._data[anon_id] = dict(record)


class JSONFileScanStore:
    """File-backed JSON dict store. Interim (see module docstring)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Guards against races WITHIN a single process only. Cross-process races
        # remain possible — that is the documented interim limitation.
        self._lock = threading.Lock()

    def _load(self) -> dict[str, dict]:
        if not self._path.exists():
            return {}
        try:
            with self._path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (ValueError, OSError):
            # Corrupt file is treated as empty rather than failing the request.
            return {}

    def get(self, anon_id: str) -> dict | None:
        with self._lock:
            rec = self._load().get(anon_id)
            return dict(rec) if rec is not None else None

    def set(self, anon_id: str, record: dict) -> None:
        with self._lock:
            data = self._load()
            data[anon_id] = dict(record)
            tmp = self._path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp.replace(self._path)

"""Async ranking job store (Phase R5).

File-backed JSON store, deliberately consistent with R3's freemium store — the one
established lightweight-persistence pattern in this no-database codebase. Tracks
each batch-ranking job's lifecycle so a recruiter can submit a large batch, get a
``job_id`` immediately, and poll for progress + the final result.

INTERIM-STORAGE FLAG (same honesty standard as R3): completed job RESULTS survive a
restart (they're on disk), but jobs that are mid-``processing`` when the single
worker process dies are orphaned — they won't resume. A stale-job reaper marks such
jobs ``failed`` on read (see ``batch_ranking``). Adequate for single-worker
free-tier hosting; not a distributed durable queue.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JobStatus:
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"

    TERMINAL = frozenset({COMPLETE, FAILED})


class RankJobStore:
    """Persist and mutate batch-ranking job records keyed by job_id."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _load(self) -> dict[str, dict]:
        if not self._path.exists():
            return {}
        try:
            with self._path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (ValueError, OSError):
            return {}

    def _save(self, data: dict[str, dict]) -> None:
        tmp = self._path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self._path)

    def create(self, job_id: str, owner_account_id: str, total: int) -> dict:
        """Register a new queued job owned by ``owner_account_id``."""
        record = {
            "job_id": job_id,
            "owner_account_id": owner_account_id,
            "status": JobStatus.QUEUED,
            "total": total,
            "completed": 0,
            "result": None,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        with self._lock:
            data = self._load()
            data[job_id] = record
            self._save(data)
        return dict(record)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            rec = self._load().get(job_id)
            return dict(rec) if rec is not None else None

    def update(self, job_id: str, **fields: Any) -> None:
        """Patch a job record's fields (status/completed/result/…)."""
        with self._lock:
            data = self._load()
            rec = data.get(job_id)
            if rec is None:
                return
            rec.update(fields)
            rec["updated_at"] = datetime.now(UTC).isoformat()
            data[job_id] = rec
            self._save(data)

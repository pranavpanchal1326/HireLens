"""Database session placeholder.

Phase 0.1 intentionally ships NO database logic. This module only records the
configured connection target so future phases have a single, obvious place to
introduce the real engine/session machinery (SQLAlchemy against SQLite locally,
or Supabase/Postgres in production).
"""

from __future__ import annotations

from app.core.config import settings

# The connection string future phases will build the engine from.
DATABASE_URL: str = settings.DATABASE_URL

"""Shared skill-text normalization (extracted in Phase 3.3).

Lives in its own dependency-free module so every layer that compares skill text
(taxonomy ingestion 3.1, the skill matcher 3.3, and the future orchestrator) can
import it WITHOUT pulling in pandas via taxonomy_ingestion. taxonomy_ingestion
re-exports it, so existing imports keep working.

Reconciliation with Phase 1.2: SkillExtractor matches via spaCy
PhraseMatcher(attr="LOWER") — case-insensitive lowercasing, no punctuation
handling. This function is a compatible SUPERSET: it lowercases (agreeing with
LOWER matching), collapses whitespace, and standardizes dash variants to a plain
hyphen, which never disagrees with LOWER matching on plain-ASCII seed skills and
only adds robustness for messier ESCO labels.
"""

from __future__ import annotations

import re

_DASH_RE = re.compile(r"[‐-―−]")
_WS_RE = re.compile(r"\s+")


def normalize_skill_text(text: str) -> str:
    """Canonical normalization used EVERYWHERE skill text is compared."""
    lowered = text.lower()
    dashed = _DASH_RE.sub("-", lowered)
    return _WS_RE.sub(" ", dashed).strip()

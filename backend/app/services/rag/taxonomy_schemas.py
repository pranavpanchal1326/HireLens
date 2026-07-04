"""Skill taxonomy data contracts (Phase 3.1).

Canonical structured representation of a skill concept ingested from ESCO. This
preserves enough structure (preferred label vs alt labels vs source) for the
downstream RAG matcher (3.3) to distinguish exact vs synonym vs purely-semantic
matches, which the UI must render differently (Design Blueprint §10.6).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SkillTaxonomyEntry(BaseModel):
    """One canonical skill concept from a taxonomy source (ESCO).

    ``concept_uri`` is the canonical identity everything downstream references.
    ``preferred_label`` is the display name; ``alt_labels`` are its synonyms —
    both kept in their source casing so exact-match detection can normalize them
    consistently via the shared ``normalize_skill_text`` at compare time.
    """

    concept_uri: str
    preferred_label: str
    alt_labels: list[str] = Field(default_factory=list)
    description: str | None = None
    skill_type: str | None = None
    source: Literal["esco"] = "esco"

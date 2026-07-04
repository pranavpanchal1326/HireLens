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


class SkillVectorEntry(BaseModel):
    """Metadata payload for one FAISS vector (Phase 3.2).

    FAISS stores only vectors + integer positions, so this parallel structure
    maps each vector back to its skill. Preferred label AND each alt label get
    their own vector (all sharing a concept_uri), so retrieval can match any known
    phrasing of a skill — ``label_type`` preserves the exact-vs-synonym distinction
    (Design Blueprint §10.6). INVARIANT: metadata list position N corresponds
    exactly to FAISS index position N.
    """

    vector_id: int
    concept_uri: str
    matched_text: str
    label_type: Literal["preferred", "alt"]

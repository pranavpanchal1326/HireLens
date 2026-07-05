"""Privacy / anonymization data contracts (Phase 9.2).

Shapes for the "blind mode" anonymization feature (PRD §9: "strip name/photo/
university before scoring — doubles as the bias-check feature"; Design Blueprint
§11.3 / §13: a VISIBLE feature that "shows what was stripped and why").

Design rule honored here: the disclosure surface must be renderable WITHOUT
re-exposing the removed PII. Every stripped item therefore carries only its
category, a redacted placeholder, and a salted hash of the original snippet —
never the original plaintext. This lets the frontend say "we removed a
university name here" without ever holding the name.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class StrippedItemType(str, Enum):
    """Category of a redacted item. Rendered as the reason shown to the user."""

    GENDER_TERM = "gender_term"
    INSTITUTION = "institution"
    PHOTO = "photo"


class StrippedItem(BaseModel):
    """One redaction, disclosable without re-exposing the original PII.

    ``original_snippet_hash`` is a salted SHA-256 hash of the removed text — it
    exists for auditing/dedup, NOT for recovery. ``replaced_with`` is the visible
    placeholder that took the original's place in the scored text.
    """

    type: StrippedItemType
    replaced_with: str
    original_snippet_hash: str = Field(
        ...,
        description="Salted SHA-256 of the removed snippet. Not reversible; never the raw value.",
    )
    occurrences: int = Field(default=1, ge=1)


class AnonymizationReport(BaseModel):
    """Result of anonymizing a piece of scored free-text.

    ``anonymized_text`` is what actually goes into scoring. ``stripped_items`` is
    the disclosure feed for the "what we stripped and why" panel.
    """

    anonymized_text: str
    stripped_items: list[StrippedItem] = Field(default_factory=list)

    @property
    def anything_stripped(self) -> bool:
        return bool(self.stripped_items)

"""Anonymization / "blind mode" service (Phase 9.2).

Implements the reality-scoped slice of PRD §9's "strip name/photo/university
before scoring" feature.

SCOPE NOTE (honest, per the bias-harness precedent in
``services/evaluation/bias_name_swap_harness.py``):
  The parser is ALREADY name-blind by design — ``ParsedResume`` never stores the
  candidate name (only ``contact_info_present: bool``). So "strip the name" is
  largely satisfied upstream and is NOT re-done here. What this service removes
  from the scored free-text is the identity signal that CAN still leak into
  scoring:
    - gender-coded terms (he/she, mr/ms, chairman/chairwoman, ...)
    - university / institution names
    - a photo marker, if the caller reports one (photos live in the source file,
      not in text; the ``photo_present`` flag lets the report reflect that)

DESIGN CHOICE — auditable wordlists/regex, NOT an opaque model:
  Per PRD §9 ("doubles as the bias-check feature") and the Design Blueprint's
  transparency principle, redaction rules must be reviewable. Gender terms come
  from an explicit frozenset; institutions from an explicit keyword pattern. No
  LLM, no hidden model decides what counts as PII here. Deterministic in →
  deterministic out, so it is unit-testable and reproducible.

Documented limitations:
  - Institution detection is keyword-anchored ("University of X", "X College")
    and will miss institutions named without such an anchor (e.g. "MIT"). A
    known-institutions list could be layered in later; flagged, not silently
    assumed solved.
  - Gender-term stripping is lexical; it does not resolve gendered names or
    pronoun coreference.

What this service does NOT do: it does not persist anything, does not touch the
trained model (Phase 6), the RAG matcher (Phase 3), or auth (Phase 9.1), and does
not run the bias name-swap harness (that lives in Phase 5.5 and stays there).
"""

from __future__ import annotations

import hashlib
import re

from app.schemas.privacy import (
    AnonymizationReport,
    StrippedItem,
    StrippedItemType,
)

# Salt for snippet hashing. Not a secret (the hash is for audit/dedup, never
# recovery), but keeping it module-local documents intent: these hashes are not
# meant to be matched against raw PII from outside this process.
_HASH_SALT = b"hirelens-anonymizer-v1"

_GENDER_PLACEHOLDER = "[REDACTED_GENDER]"
_INSTITUTION_PLACEHOLDER = "[REDACTED_INSTITUTION]"

# Explicit, reviewable gender-coded term set. Lowercased; matched whole-word,
# case-insensitively. Deliberately conservative — high-signal identity terms only.
_GENDER_TERMS: frozenset[str] = frozenset({
    "he", "him", "his", "she", "her", "hers",
    "mr", "mrs", "ms", "miss", "sir", "madam",
    "male", "female", "man", "woman",
    "husband", "wife", "boyfriend", "girlfriend",
    "father", "mother", "son", "daughter",
    "chairman", "chairwoman", "businessman", "businesswoman",
})

# Whole-word, case-insensitive gender-term matcher built once from the set.
_GENDER_RE = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, _GENDER_TERMS), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Institution matcher. Anchors on common higher-education keywords and grabs the
# proximate proper-noun span so "University of Toronto" / "Stanford University" /
# "Boston College" are removed as a unit. Keyword-anchored by design (see
# limitations above).
_INSTITUTION_RE = re.compile(
    r"\b("
    r"(?:[A-Z][\w.&'-]+\s+){0,3}(?:University|College|Institute|Polytechnic)"
    r"(?:\s+of\s+(?:[A-Z][\w.&'-]+)(?:\s+[A-Z][\w.&'-]+){0,2})?"
    r"|University\s+of\s+(?:[A-Z][\w.&'-]+)(?:\s+[A-Z][\w.&'-]+){0,2}"
    r")\b"
)


def _hash_snippet(snippet: str) -> str:
    """Salted, non-reversible SHA-256 of a removed snippet (normalized)."""
    normalized = snippet.strip().lower().encode("utf-8")
    return hashlib.sha256(_HASH_SALT + normalized).hexdigest()


def _redact(
    text: str,
    pattern: re.Pattern[str],
    placeholder: str,
    item_type: StrippedItemType,
) -> tuple[str, list[StrippedItem]]:
    """Replace every match of ``pattern`` with ``placeholder``.

    Groups identical (normalized) snippets into a single StrippedItem with an
    occurrence count, so the disclosure feed is deduplicated.
    """
    counts: dict[str, int] = {}

    def _sub(match: re.Match[str]) -> str:
        key = _hash_snippet(match.group(0))
        counts[key] = counts.get(key, 0) + 1
        return placeholder

    redacted = pattern.sub(_sub, text)
    items = [
        StrippedItem(
            type=item_type,
            replaced_with=placeholder,
            original_snippet_hash=key,
            occurrences=n,
        )
        for key, n in counts.items()
    ]
    return redacted, items


def anonymize_text(text: str, photo_present: bool = False) -> AnonymizationReport:
    """Strip identity signals from scored free-text and report what was removed.

    Order is fixed (institution before gender terms) so the result is fully
    deterministic. Returns the anonymized text plus a disclosure feed that never
    contains the original PII — only category, placeholder, and a salted hash.

    Args:
        text: The free-text to anonymize (e.g. a resume's scored text).
        photo_present: Whether the source document carried a photo. Photos are not
            in ``text``; this flag lets the report disclose the removal honestly.

    Returns:
        AnonymizationReport with ``anonymized_text`` and ``stripped_items``.
    """
    stripped: list[StrippedItem] = []

    working, institution_items = _redact(
        text, _INSTITUTION_RE, _INSTITUTION_PLACEHOLDER, StrippedItemType.INSTITUTION
    )
    stripped.extend(institution_items)

    working, gender_items = _redact(
        working, _GENDER_RE, _GENDER_PLACEHOLDER, StrippedItemType.GENDER_TERM
    )
    stripped.extend(gender_items)

    if photo_present:
        stripped.append(
            StrippedItem(
                type=StrippedItemType.PHOTO,
                replaced_with="[REDACTED_PHOTO]",
                # No text snippet exists for a photo; hash a stable marker so the
                # field contract (always a hash) still holds.
                original_snippet_hash=_hash_snippet("<photo>"),
                occurrences=1,
            )
        )

    return AnonymizationReport(anonymized_text=working, stripped_items=stripped)

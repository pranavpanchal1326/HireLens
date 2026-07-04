"""ESCO taxonomy ingestion (Phase 3.1).

Transforms the raw ESCO skills CSV export into a clean, structured, queryable
list of ``SkillTaxonomyEntry`` objects. This is the vocabulary foundation of the
RAG Skill Matcher (PRD §4). Sloppy parsing here silently corrupts every
downstream retrieval, so parsing is defensive, deterministic, and tested.

Does NOT embed skills (Part 3.2), match (3.3), or touch seed_skills.txt (1.2).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from app.services.rag.taxonomy_schemas import SkillTaxonomyEntry

logger = logging.getLogger(__name__)

# ESCO skills export column names (verified against the on-disk file).
_COL_URI = "conceptUri"
_COL_PREFERRED = "preferredLabel"
_COL_ALT = "altLabels"
_COL_TYPE = "skillType"
_COL_DESCRIPTION = "description"
_REQUIRED_COLUMNS = (_COL_URI, _COL_PREFERRED, _COL_ALT)

# altLabels within one cell are newline-delimited in the current ESCO export, but
# older/other exports use pipes — split on either, robustly.
_ALT_DELIMITER_RE = re.compile(r"[|\r\n]+")
# Normalization: collapse any dash variant to a plain hyphen before comparing.
_DASH_RE = re.compile(r"[‐-―−]")
_WS_RE = re.compile(r"\s+")


class MalformedTaxonomySourceError(Exception):
    """Raised when the ESCO source file lacks expected columns/structure."""


def normalize_skill_text(text: str) -> str:
    """Canonical skill-text normalization used EVERYWHERE skills are compared.

    Reconciliation with Phase 1.2: SkillExtractor matches via spaCy
    PhraseMatcher(attr="LOWER"), i.e. case-insensitive lowercasing with no
    punctuation handling. This function is a compatible SUPERSET — it lowercases
    (so it agrees with LOWER matching), collapses whitespace, and standardizes
    dash variants to a plain hyphen so "e-commerce" / "e—commerce" compare equal.
    The extra dash/whitespace handling never disagrees with LOWER matching on the
    plain ASCII skills the seed vocabulary uses; it only adds robustness for the
    messier ESCO labels.
    """
    lowered = text.lower()
    dashed = _DASH_RE.sub("-", lowered)
    return _WS_RE.sub(" ", dashed).strip()


class ESCOTaxonomyIngester:
    """Loads and cleans the raw ESCO skills CSV into SkillTaxonomyEntry objects."""

    def __init__(self, raw_csv_path: str) -> None:
        self.raw_csv_path = raw_csv_path
        self.last_skipped_count = 0

    def load_raw(self) -> pd.DataFrame:
        """Load the ESCO CSV, validating that expected columns are present.

        Empty cells are read as "" (not NaN) so downstream string handling is
        uniform. Missing required columns raise a clear, typed error rather than
        letting a bare pandas KeyError surface later during parsing.
        """
        try:
            df = pd.read_csv(self.raw_csv_path, dtype=str, keep_default_na=False)
        except (OSError, ValueError) as exc:
            raise MalformedTaxonomySourceError(
                f"Could not read ESCO CSV at {self.raw_csv_path}: {exc}"
            ) from exc

        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise MalformedTaxonomySourceError(
                f"ESCO source is missing required column(s): {missing}. "
                f"Found columns: {list(df.columns)}"
            )
        return df

    def parse_alt_labels(self, raw_alt_label_field: str) -> list[str]:
        """Split ESCO's alt-label cell into a clean synonym list.

        Handles BOTH newline- and pipe-delimited cells, strips whitespace, drops
        empties, and deduplicates case-insensitively while preserving the
        first-seen casing (so display stays natural, but no synonym repeats).
        """
        if not raw_alt_label_field:
            return []
        result: list[str] = []
        seen: set[str] = set()
        for piece in _ALT_DELIMITER_RE.split(raw_alt_label_field):
            label = piece.strip()
            if not label:
                continue
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(label)
        return result

    def ingest(self) -> list[SkillTaxonomyEntry]:
        """Load, clean, and return all skill entries. Skips (and counts) rows with
        a missing concept_uri or preferred_label, and de-dupes repeated URIs."""
        df = self.load_raw()
        entries: list[SkillTaxonomyEntry] = []
        seen_uris: set[str] = set()
        skipped = 0

        for row in df.itertuples(index=False):
            uri = str(getattr(row, _COL_URI, "") or "").strip()
            preferred = str(getattr(row, _COL_PREFERRED, "") or "").strip()
            if not uri or not preferred or uri in seen_uris:
                skipped += 1
                continue
            seen_uris.add(uri)

            skill_type = str(getattr(row, _COL_TYPE, "") or "").strip() or None
            description = str(getattr(row, _COL_DESCRIPTION, "") or "").strip() or None
            alt_labels = self.parse_alt_labels(str(getattr(row, _COL_ALT, "") or ""))
            entries.append(
                SkillTaxonomyEntry(
                    concept_uri=uri,
                    preferred_label=preferred,
                    alt_labels=alt_labels,
                    description=description,
                    skill_type=skill_type,
                    source="esco",
                )
            )

        self.last_skipped_count = skipped
        if skipped:
            logger.warning("Skipped %d ESCO rows (missing/duplicate key)", skipped)
        return entries


def save_taxonomy(entries: list[SkillTaxonomyEntry], output_path: str) -> None:
    """Persist entries as JSON Lines (one entry per line).

    JSON Lines is chosen over pickle because a taxonomy is something you WANT to
    spot-check by eye during development — it stays human-inspectable and diffable,
    and round-trips Pydantic models cleanly.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry.model_dump_json() + "\n")


def load_taxonomy(input_path: str) -> list[SkillTaxonomyEntry]:
    """Load entries previously persisted by ``save_taxonomy``."""
    entries: list[SkillTaxonomyEntry] = []
    with Path(input_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(SkillTaxonomyEntry.model_validate_json(line))
    return entries

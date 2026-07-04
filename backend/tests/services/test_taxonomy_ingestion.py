"""Tests for the Phase 3.1 ESCO taxonomy ingestion."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.services.rag.taxonomy_ingestion import (
    ESCOTaxonomyIngester,
    MalformedTaxonomySourceError,
    load_taxonomy,
    normalize_skill_text,
    save_taxonomy,
)

_COLUMNS = ["conceptUri", "preferredLabel", "altLabels", "skillType", "description"]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@pytest.fixture
def good_csv(tmp_path: Path) -> Path:
    path = tmp_path / "skills.csv"
    _write_csv(
        path,
        [
            {
                "conceptUri": "esco/skill/1",
                "preferredLabel": "people management",
                # newline-delimited alt labels (real ESCO style), with a dup casing.
                "altLabels": "team leadership\nManaging People\nteam leadership",
                "skillType": "skill/competence",
                "description": "Managing a team of people.",
            },
            {
                "conceptUri": "esco/skill/2",
                "preferredLabel": "python programming",
                # pipe-delimited alt labels (older export style).
                "altLabels": "python|python development|coding in python",
                "skillType": "knowledge",
                "description": "",
            },
            {  # empty preferred_label → must be skipped.
                "conceptUri": "esco/skill/3",
                "preferredLabel": "",
                "altLabels": "orphan",
                "skillType": "",
                "description": "",
            },
        ],
    )
    return path


def test_parses_well_formed_row(good_csv: Path) -> None:
    entries = ESCOTaxonomyIngester(str(good_csv)).ingest()
    by_uri = {e.concept_uri: e for e in entries}
    entry = by_uri["esco/skill/1"]
    assert entry.preferred_label == "people management"
    assert entry.skill_type == "skill/competence"
    assert entry.source == "esco"
    assert "team leadership" in entry.alt_labels


def test_alt_labels_pipe_delimited() -> None:
    ing = ESCOTaxonomyIngester("unused")
    assert ing.parse_alt_labels("a|b|c") == ["a", "b", "c"]


def test_alt_labels_newline_delimited() -> None:
    ing = ESCOTaxonomyIngester("unused")
    assert ing.parse_alt_labels("a\nb\r\nc") == ["a", "b", "c"]


def test_alt_labels_dedupe_case_insensitive_preserving_casing() -> None:
    ing = ESCOTaxonomyIngester("unused")
    result = ing.parse_alt_labels("Team Lead|team lead|TEAM LEAD|Scrum")
    assert result == ["Team Lead", "Scrum"]  # first-seen casing kept, dups removed


def test_normalize_skill_text_pairs() -> None:
    assert normalize_skill_text("  Machine   Learning ") == "machine learning"
    assert normalize_skill_text("E‑Commerce") == "e-commerce"  # unicode dash → hyphen
    assert normalize_skill_text("PYTHON") == "python"
    assert normalize_skill_text("Data\tScience") == "data science"


def test_missing_required_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["conceptUri", "preferredLabel"])
        writer.writeheader()
        writer.writerow({"conceptUri": "x", "preferredLabel": "y"})
    with pytest.raises(MalformedTaxonomySourceError) as exc:
        ESCOTaxonomyIngester(str(path)).ingest()
    assert "altLabels" in str(exc.value)  # names the missing column


def test_empty_preferred_label_rows_skipped_and_counted(good_csv: Path) -> None:
    ing = ESCOTaxonomyIngester(str(good_csv))
    entries = ing.ingest()
    uris = {e.concept_uri for e in entries}
    assert "esco/skill/3" not in uris  # the empty-label row is dropped
    assert ing.last_skipped_count == 1  # and observably counted


def test_save_load_roundtrip_identical(good_csv: Path, tmp_path: Path) -> None:
    entries = ESCOTaxonomyIngester(str(good_csv)).ingest()
    out = tmp_path / "tax.jsonl"
    save_taxonomy(entries, str(out))
    reloaded = load_taxonomy(str(out))
    assert len(reloaded) == len(entries)
    for original, restored in zip(entries, reloaded, strict=True):
        assert original.model_dump() == restored.model_dump()

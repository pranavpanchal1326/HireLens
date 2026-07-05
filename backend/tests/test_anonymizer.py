# ruff: noqa: E501
"""Tests for the Phase 9.2 anonymization / blind-mode service."""

from __future__ import annotations

from app.schemas.privacy import StrippedItemType
from app.services.privacy.anonymizer import anonymize_text


def test_gender_terms_stripped_whole_word() -> None:
    """Gender-coded terms are redacted; substrings inside other words are left alone."""
    report = anonymize_text("He led the team. The chairman spoke. Theremin player.")
    assert "He" not in report.anonymized_text
    assert "chairman" not in report.anonymized_text
    # "Theremin" contains "he"/"her" as substrings but must survive (whole-word).
    assert "Theremin" in report.anonymized_text
    types = {item.type for item in report.stripped_items}
    assert StrippedItemType.GENDER_TERM in types


def test_institution_stripped() -> None:
    """University/college names are redacted as a unit."""
    report = anonymize_text("Studied at University of Toronto and later Boston College.")
    assert "Toronto" not in report.anonymized_text
    assert "Boston College" not in report.anonymized_text
    assert "[REDACTED_INSTITUTION]" in report.anonymized_text
    assert any(i.type == StrippedItemType.INSTITUTION for i in report.stripped_items)


def test_report_never_contains_raw_pii() -> None:
    """The disclosure feed exposes only category + placeholder + hash, never the PII."""
    report = anonymize_text("She graduated from Stanford University.")
    for item in report.stripped_items:
        assert "Stanford" not in item.original_snippet_hash
        assert "Stanford" not in item.replaced_with
        assert len(item.original_snippet_hash) == 64  # sha256 hexdigest


def test_deterministic_output() -> None:
    """Same input yields identical anonymized text and identical hashes."""
    text = "He studied at University of Oxford."
    a = anonymize_text(text)
    b = anonymize_text(text)
    assert a.anonymized_text == b.anonymized_text
    assert [i.original_snippet_hash for i in a.stripped_items] == [
        i.original_snippet_hash for i in b.stripped_items
    ]


def test_repeated_term_deduplicated_with_count() -> None:
    """Identical stripped snippets collapse into one item with an occurrence count."""
    report = anonymize_text("He is here. He left. He returned.")
    gender_items = [i for i in report.stripped_items if i.type == StrippedItemType.GENDER_TERM]
    he_items = [i for i in gender_items if i.occurrences >= 3]
    assert he_items, "expected the repeated 'He' to be counted, not duplicated"


def test_photo_flag_disclosed() -> None:
    """A reported photo yields a PHOTO stripped item even though text has no photo."""
    report = anonymize_text("Experienced engineer.", photo_present=True)
    assert any(i.type == StrippedItemType.PHOTO for i in report.stripped_items)


def test_clean_text_strips_nothing() -> None:
    """Text with no identity signals passes through unchanged with an empty feed."""
    text = "Built scalable data pipelines using python and airflow."
    report = anonymize_text(text)
    assert report.anonymized_text == text
    assert report.stripped_items == []
    assert report.anything_stripped is False

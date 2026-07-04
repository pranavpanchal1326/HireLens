"""Tests for the Phase 3.2 FAISS skill vector store.

Uses a small synthetic taxonomy + the real embedding model (loaded once), so the
critical semantic-retrieval test exercises genuine embeddings, not a stub.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.rag.faiss_index_builder import (
    FAISSSkillIndexBuilder,
    FAISSSkillIndexQuerier,
    IndexMetadataMismatchError,
    load_index,
    save_index,
)
from app.services.rag.taxonomy_schemas import SkillTaxonomyEntry
from app.services.scoring.embedding_scorer import EmbeddingScorer

_SCORER = EmbeddingScorer()

FIXTURE_ENTRIES = [
    SkillTaxonomyEntry(
        concept_uri="esco/1",
        preferred_label="people management",
        alt_labels=["team leadership", "managing staff"],
    ),
    SkillTaxonomyEntry(
        concept_uri="esco/2",
        preferred_label="python programming",
        alt_labels=["python development", "coding in python"],
    ),
    SkillTaxonomyEntry(
        concept_uri="esco/3",
        preferred_label="financial accounting",
        alt_labels=["bookkeeping"],
    ),
    SkillTaxonomyEntry(
        concept_uri="esco/4",
        preferred_label="graphic design",
        alt_labels=["visual design", "branding"],
    ),
]

# Total vectors = sum(1 preferred + len(alt_labels)) = (1+2)+(1+2)+(1+1)+(1+2) = 11
_EXPECTED_VECTORS = 11


@pytest.fixture(scope="module")
def built() -> tuple[object, list]:
    builder = FAISSSkillIndexBuilder(_SCORER)
    index, metadata = builder.build_index(FIXTURE_ENTRIES)
    return index, metadata


def test_index_size_matches_total_labels(built: tuple) -> None:
    index, metadata = built
    assert index.ntotal == _EXPECTED_VECTORS
    assert len(metadata) == _EXPECTED_VECTORS


def test_label_types_are_correct(built: tuple) -> None:
    _, metadata = built
    preferred = [m for m in metadata if m.label_type == "preferred"]
    alt = [m for m in metadata if m.label_type == "alt"]
    assert len(preferred) == 4  # one per concept
    assert len(alt) == 7
    # A preferred entry maps to its canonical label.
    people = next(m for m in metadata if m.matched_text == "people management")
    assert people.label_type == "preferred"
    assert people.concept_uri == "esco/1"


def test_semantic_retrieval_no_exact_overlap(built: tuple) -> None:
    """CRITICAL: a phrase sharing NO words with any label retrieves the right
    concept with a meaningfully high score."""
    index, metadata = built
    querier = FAISSSkillIndexQuerier(index, metadata, _SCORER)
    results = querier.query_raw("led a team of engineers", top_k=3)

    top_uris = {entry.concept_uri for entry, _ in results}
    assert "esco/1" in top_uris  # people management / team leadership concept
    # The best-matching vector for this concept should score meaningfully high.
    best_for_concept = max(
        score for entry, score in results if entry.concept_uri == "esco/1"
    )
    assert best_for_concept >= 0.35


def test_exact_query_scores_near_one(built: tuple) -> None:
    """Querying with a label already in the index returns it at ~1.0 cosine —
    proves L2-normalization + inner product reproduces cosine similarity."""
    index, metadata = built
    querier = FAISSSkillIndexQuerier(index, metadata, _SCORER)
    results = querier.query_raw("python programming", top_k=1)
    top_entry, top_score = results[0]
    assert top_entry.matched_text == "python programming"
    assert top_score >= 0.99


def test_save_load_roundtrip_identical_results(built: tuple, tmp_path: Path) -> None:
    index, metadata = built
    idx_path = str(tmp_path / "skill.index")
    meta_path = str(tmp_path / "skill_meta.jsonl")
    save_index(index, metadata, idx_path, meta_path)

    r_index, r_metadata = load_index(idx_path, meta_path)
    original = FAISSSkillIndexQuerier(index, metadata, _SCORER).query_raw(
        "staff management", top_k=3
    )
    reloaded = FAISSSkillIndexQuerier(r_index, r_metadata, _SCORER).query_raw(
        "staff management", top_k=3
    )
    assert [(e.vector_id, round(s, 5)) for e, s in original] == [
        (e.vector_id, round(s, 5)) for e, s in reloaded
    ]


def test_mismatched_metadata_raises(built: tuple, tmp_path: Path) -> None:
    index, metadata = built
    idx_path = str(tmp_path / "skill.index")
    meta_path = tmp_path / "skill_meta.jsonl"
    save_index(index, metadata, idx_path, str(meta_path))

    # Truncate the metadata file: drop the last two lines to force a mismatch.
    lines = meta_path.read_text(encoding="utf-8").splitlines()
    meta_path.write_text("\n".join(lines[:-2]) + "\n", encoding="utf-8")

    with pytest.raises(IndexMetadataMismatchError):
        load_index(idx_path, str(meta_path))

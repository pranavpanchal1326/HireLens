"""Minimal tests for the Phase 3.4 RAG similar-case lookup."""

from __future__ import annotations

from app.services.rag.rag_similar_case_lookup import (
    OUTLIER_DEVIATION_THRESHOLD,
    SimilarCaseStore,
    StoredCase,
    make_case_id,
    utc_now_iso,
)
from app.services.scoring.embedding_scorer import EmbeddingScorer

_SCORER = EmbeddingScorer()


def _case(score: int) -> StoredCase:
    return StoredCase(
        case_id=make_case_id(),
        final_score=score,
        confidence_level="medium",
        pipeline_version="v3-hybrid",
        timestamp=utc_now_iso(),
    )


def _store() -> SimilarCaseStore:
    return SimilarCaseStore(_SCORER)


def test_normal_retrieval_returns_nearest_first() -> None:
    store = _store()
    # A "python backend" case and a "graphic design" case.
    py_emb = store.build_case_embedding("python backend engineer", "python api role")
    gd_emb = store.build_case_embedding(
        "graphic designer branding", "visual design job"
    )
    store.store_case(_case(80), py_emb)
    store.store_case(_case(40), gd_emb)

    query = store.build_case_embedding(
        "senior python developer", "backend python position"
    )
    results = store.retrieve_similar_cases(query, k=2)
    assert len(results) == 2
    # The python case should be the nearest neighbor.
    assert results[0][1] >= results[1][1]
    assert results[0][0].final_score == 80


def test_cold_start_sets_low_sample_warning() -> None:
    store = _store()
    store.store_case(_case(70), store.build_case_embedding("a resume", "a jd"))
    query = store.build_case_embedding("another resume", "another jd")
    similar = store.retrieve_similar_cases(query, k=5)
    assert len(similar) == 1  # only what exists, not padded
    result = store.calibration_check(72, similar, requested_k=5)
    assert result.low_sample_warning is True
    assert result.is_outlier is False  # never assert outlier on thin evidence


def test_empty_store_returns_nothing_and_warns() -> None:
    store = _store()
    query = store.build_case_embedding("resume", "jd")
    similar = store.retrieve_similar_cases(query, k=5)
    assert similar == []
    result = store.calibration_check(50, similar, requested_k=5)
    assert result.low_sample_warning is True
    assert result.is_outlier is False
    assert result.deviation == 0.0


def test_outlier_flag_triggers_on_deviation() -> None:
    store = _store()
    # Five consistent ~80 cases (enough sample), varied text so vectors differ.
    for i in range(5):
        emb = store.build_case_embedding(f"python engineer variant {i}", "python role")
        store.store_case(_case(80), emb)
    query = store.build_case_embedding("python engineer new", "python role")
    similar = store.retrieve_similar_cases(query, k=5)

    # A wildly low new score vs a median of 80 → outlier.
    outlier = store.calibration_check(20, similar, requested_k=5)
    assert outlier.deviation > OUTLIER_DEVIATION_THRESHOLD
    assert outlier.is_outlier is True

    # A consistent new score → not an outlier.
    consistent = store.calibration_check(82, similar, requested_k=5)
    assert consistent.is_outlier is False
    assert consistent.similar_case_scores == [80, 80, 80, 80, 80]


def test_persistence_roundtrip(tmp_path: str) -> None:
    idx = str(tmp_path) + "/cases.index"
    meta = str(tmp_path) + "/cases.jsonl"
    store = SimilarCaseStore(_SCORER, index_path=idx, metadata_path=meta)
    store.store_case(_case(65), store.build_case_embedding("resume text", "jd text"))

    reloaded = SimilarCaseStore(_SCORER, index_path=idx, metadata_path=meta)
    query = reloaded.build_case_embedding("resume text", "jd text")
    similar = reloaded.retrieve_similar_cases(query, k=1)
    assert len(similar) == 1
    assert similar[0][0].final_score == 65

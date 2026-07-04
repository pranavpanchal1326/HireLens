"""Tests for the Phase 2.3 embedding cache layer.

Uses a lightweight fake EmbeddingScorer (deterministic hash-based vectors) so
tests are fast and can assert exact call counts — no real transformer model
needed to prove the caching mechanics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services.scoring.embedding_cache import CachedEmbeddingScorer, EmbeddingCache
from app.services.scoring.embedding_scorer import EmbeddingScorer


class SpyEmbeddingScorer:
    """Deterministic stand-in that records how it was called."""

    def __init__(self) -> None:
        self.embed_calls: list[str] = []
        self.embed_batch_calls: list[list[str]] = []

    def _vec(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.standard_normal(8).astype(np.float32)

    def embed(self, text: str) -> np.ndarray:
        self.embed_calls.append(text)
        return self._vec(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        self.embed_batch_calls.append(list(texts))
        return np.stack([self._vec(t) for t in texts])


@pytest.fixture
def cache(tmp_path: Path) -> EmbeddingCache:
    return EmbeddingCache(cache_dir=str(tmp_path / "cache"))


# --- EmbeddingCache ----------------------------------------------------------


def test_set_then_get_roundtrip(cache: EmbeddingCache) -> None:
    vec = np.arange(8, dtype=np.float32)
    cache.set("doc-1", vec)
    loaded = cache.get("doc-1")
    assert loaded is not None
    assert np.array_equal(loaded, vec)


def test_get_missing_returns_none(cache: EmbeddingCache) -> None:
    assert cache.get("nope") is None


def test_has_reflects_state(cache: EmbeddingCache) -> None:
    assert cache.has("d") is False
    cache.set("d", np.zeros(4, dtype=np.float32))
    assert cache.has("d") is True


def test_invalidate_removes_entry(cache: EmbeddingCache) -> None:
    cache.set("d", np.ones(4, dtype=np.float32))
    cache.invalidate("d")
    assert cache.has("d") is False
    assert cache.get("d") is None


def test_corrupt_cache_file_is_graceful_miss(cache: EmbeddingCache) -> None:
    # Write garbage where a .npy would live.
    path = cache._path("broken")
    path.write_bytes(b"this is not a valid npy file")
    assert cache.get("broken") is None  # graceful miss, no exception


def test_distinct_ids_do_not_collide(cache: EmbeddingCache) -> None:
    # "a.b" and "a_b" both sanitize to "a_b"; the id hash must keep them distinct.
    cache.set("a.b", np.full(4, 1.0, dtype=np.float32))
    cache.set("a_b", np.full(4, 2.0, dtype=np.float32))
    first = cache.get("a.b")
    second = cache.get("a_b")
    assert first is not None and second is not None
    assert not np.array_equal(first, second)


def test_stats_report_counts_and_size(cache: EmbeddingCache) -> None:
    cache.set("a", np.zeros(16, dtype=np.float32))
    cache.set("b", np.zeros(16, dtype=np.float32))
    stats = cache.stats()
    assert stats["total_cached_documents"] == 2
    assert stats["cache_size_mb"] >= 0.0


# --- CachedEmbeddingScorer ---------------------------------------------------


@pytest.fixture
def cached_scorer(
    cache: EmbeddingCache,
) -> tuple[CachedEmbeddingScorer, SpyEmbeddingScorer]:
    spy = SpyEmbeddingScorer()
    wrapper = CachedEmbeddingScorer(spy, cache)  # type: ignore[arg-type]
    return wrapper, spy


def test_single_embed_computed_once_then_cached(
    cached_scorer: tuple[CachedEmbeddingScorer, SpyEmbeddingScorer],
) -> None:
    wrapper, spy = cached_scorer
    first = wrapper.get_or_compute_embedding("doc-1", "some resume text")
    second = wrapper.get_or_compute_embedding("doc-1", "some resume text")
    assert len(spy.embed_calls) == 1  # second call served from cache
    assert np.array_equal(first, second)


def test_batch_only_embeds_uncached_texts(
    cached_scorer: tuple[CachedEmbeddingScorer, SpyEmbeddingScorer],
) -> None:
    """CRITICAL: prove embed_batch runs on ONLY the uncached texts."""
    wrapper, spy = cached_scorer
    # Pre-cache 3 documents.
    for i in range(3):
        wrapper.get_or_compute_embedding(f"cached-{i}", f"text {i}")
    spy.embed_calls.clear()

    ids = ["cached-0", "cached-1", "cached-2", "new-0", "new-1"]
    texts = ["text 0", "text 1", "text 2", "new text 0", "new text 1"]
    wrapper.get_or_compute_embeddings_batch(ids, texts)

    assert len(spy.embed_batch_calls) == 1
    assert spy.embed_batch_calls[0] == ["new text 0", "new text 1"]  # only the 2 new


def test_batch_preserves_input_order(
    cached_scorer: tuple[CachedEmbeddingScorer, SpyEmbeddingScorer],
) -> None:
    wrapper, spy = cached_scorer
    wrapper.get_or_compute_embedding("b", "text b")  # cache one out of order

    ids = ["a", "b", "c"]
    texts = ["text a", "text b", "text c"]
    results = wrapper.get_or_compute_embeddings_batch(ids, texts)

    # Each result must equal the direct embedding of its own text, in order.
    for doc_text, vec in zip(texts, results, strict=True):
        assert np.array_equal(vec, spy._vec(doc_text))


def test_score_identical_across_cache_states(
    cache: EmbeddingCache,
) -> None:
    """score() must match whether embeddings are fresh, cached, or mixed."""
    spy = SpyEmbeddingScorer()
    wrapper = CachedEmbeddingScorer(spy, cache)  # type: ignore[arg-type]

    # Fresh: nothing cached yet.
    fresh = wrapper.score("r1", "resume text", "j1", "jd text")
    # Both cached now.
    both_cached = wrapper.score("r1", "resume text", "j1", "jd text")
    # Mixed: r1 cached, new jd computed — then compare same underlying texts.
    cache.invalidate("j1")
    mixed = wrapper.score("r1", "resume text", "j1", "jd text")

    assert fresh == both_cached == mixed


def test_score_matches_uncached_reference(cache: EmbeddingCache) -> None:
    """Cache-aware score equals the direct normalized_similarity of the vectors."""
    spy = SpyEmbeddingScorer()
    wrapper = CachedEmbeddingScorer(spy, cache)  # type: ignore[arg-type]
    expected = EmbeddingScorer.normalized_similarity(
        spy._vec("resume text"), spy._vec("jd text")
    )
    actual = wrapper.score("r1", "resume text", "j1", "jd text")
    assert actual == expected

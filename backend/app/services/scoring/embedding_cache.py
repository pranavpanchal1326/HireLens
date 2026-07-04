"""Embedding cache layer (Phase 2.3).

Implements PRD §8.2: "embed each JD/resume once, store vectors, never recompute
per comparison." Sits IN FRONT of the Phase 2.2 EmbeddingScorer via composition
(not inheritance), so the scorer stays clean and independently testable.

Storage choice — a simple numpy file-per-key store, NOT FAISS. FAISS is built for
approximate nearest-neighbor SEARCH across many vectors (Phase 3's RAG problem).
This layer only needs exact "give me document X's embedding if I already have it"
lookups keyed by document_id — a key/value problem. numpy .npy files are the right
tool: native fast binary format, incremental per-key writes (no monolithic-file
rewrites), and individually inspectable for debugging, with zero DB infrastructure.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

from app.services.scoring.embedding_scorer import EmbeddingScorer

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = "data/processed/embedding_cache"
# Prefix for in-flight atomic-write temp files, so stats() can exclude a stray
# temp left behind by a crash mid-write.
_TMP_PREFIX = "_wtmp_"


def _safe_key(document_id: str) -> str:
    """Map a document_id to a unique, filesystem-safe filename stem.

    A readable sanitized prefix is kept for debuggability, but a hash of the FULL
    original id is appended so distinct ids can never collide onto the same file
    (e.g. "a.b" and "a_b" both sanitize to "a_b" — the hash keeps them distinct).
    """
    digest = hashlib.sha1(document_id.encode("utf-8")).hexdigest()[:12]
    prefix = "".join(c if (c.isalnum() or c in "-_") else "_" for c in document_id)
    return f"{prefix[:48]}_{digest}"


class EmbeddingCache:
    """Persistent on-disk cache of embeddings, one .npy file per document_id."""

    def __init__(self, cache_dir: str = _DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, document_id: str) -> Path:
        return self.cache_dir / f"{_safe_key(document_id)}.npy"

    def has(self, document_id: str) -> bool:
        """Fast existence check without loading the array."""
        return self._path(document_id).exists()

    def get(self, document_id: str) -> np.ndarray | None:
        """Return the cached embedding, or None on miss/corruption.

        A file left half-written by a crashed previous process is treated as a
        cache miss (logged), never a hard failure of the request.
        """
        path = self._path(document_id)
        if not path.exists():
            return None
        try:
            loaded: np.ndarray = np.load(path, allow_pickle=False)
            return loaded
        except (ValueError, OSError, EOFError) as exc:
            logger.warning("Corrupt cache entry %s treated as miss: %s", path, exc)
            return None

    def set(self, document_id: str, embedding: np.ndarray) -> None:
        """Persist an embedding using an atomic write (temp file + rename).

        Atomicity matters: a crash mid-write must never leave a half-written,
        corrupt .npy that a future get() would fail on. os.replace is atomic on
        the same filesystem, so a reader ever sees either the old file or the
        fully-written new one — never a partial file.
        """
        path = self._path(document_id)
        # Unique temp file in the SAME directory (so os.replace stays atomic — a
        # cross-filesystem rename is not atomic). The ".npy" suffix means np.save
        # writes to this exact path without appending a second extension.
        fd, tmp_name = tempfile.mkstemp(
            dir=self.cache_dir, prefix=_TMP_PREFIX, suffix=".npy"
        )
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            np.save(tmp, embedding, allow_pickle=False)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()

    def invalidate(self, document_id: str) -> None:
        """Remove a cached entry (e.g. resume edited → rescan; PRD §3.1).

        Prevents an edited document from silently reusing the stale embedding of
        its previous version. No-op if nothing is cached.
        """
        path = self._path(document_id)
        if path.exists():
            path.unlink()

    def stats(self) -> dict[str, float | int]:
        files = [
            f
            for f in self.cache_dir.glob("*.npy")
            if not f.name.startswith(_TMP_PREFIX)  # exclude stray write-temps
        ]
        total_bytes = sum(f.stat().st_size for f in files)
        return {
            "total_cached_documents": len(files),
            "cache_size_mb": round(total_bytes / (1024 * 1024), 4),
        }


class CachedEmbeddingScorer:
    """Cache-aware wrapper over EmbeddingScorer (composition, not inheritance)."""

    def __init__(
        self, embedding_scorer: EmbeddingScorer, cache: EmbeddingCache
    ) -> None:
        self._scorer = embedding_scorer
        self._cache = cache

    def get_or_compute_embedding(self, document_id: str, text: str) -> np.ndarray:
        """Return the cached embedding for ``document_id`` or compute + cache it.

        This is what the rest of the system should call instead of
        EmbeddingScorer.embed() directly once caching exists.
        """
        if self._cache.has(document_id):
            cached = self._cache.get(document_id)
            if cached is not None:
                return cached
        embedding = self._scorer.embed(text)
        self._cache.set(document_id, embedding)
        return embedding

    def get_or_compute_embeddings_batch(
        self, document_ids: list[str], texts: list[str]
    ) -> list[np.ndarray]:
        """Compute embeddings for a batch, embedding ONLY the uncached documents.

        Cached entries are read from disk; the genuinely-uncached texts are sent
        to embed_batch() in a SINGLE inference call (batching is much cheaper than
        N calls on CPU). Results are reassembled in the original input order. This
        is what makes recruiter batch-ranking (1 JD → N resumes) fast on
        repeat/overlapping requests.
        """
        if len(document_ids) != len(texts):
            raise ValueError("document_ids and texts must be the same length.")

        results: list[np.ndarray | None] = [None] * len(document_ids)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, doc_id in enumerate(document_ids):
            cached = self._cache.get(doc_id) if self._cache.has(doc_id) else None
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(texts[i])

        if uncached_texts:
            fresh = self._scorer.embed_batch(uncached_texts)
            for idx, vector in zip(uncached_indices, fresh, strict=True):
                results[idx] = vector
                self._cache.set(document_ids[idx], vector)

        # By construction every slot is now filled.
        return [vec for vec in results if vec is not None]

    def score(
        self, resume_id: str, resume_text: str, jd_id: str, jd_text: str
    ) -> float:
        """Cache-aware semantic score, identical to EmbeddingScorer.score()."""
        resume_vec = self.get_or_compute_embedding(resume_id, resume_text)
        jd_vec = self.get_or_compute_embedding(jd_id, jd_text)
        return EmbeddingScorer.normalized_similarity(resume_vec, jd_vec)

    def get_cache_stats(self) -> dict[str, float | int]:
        return self._cache.stats()

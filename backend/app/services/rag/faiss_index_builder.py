"""FAISS skill vector store (Phase 3.2).

Makes the ESCO taxonomy searchable BY MEANING — the mechanism that lets the
system match "led team" ≈ "people management" with no shared words (PRD §4).

Design decision (deliberate, documented): the preferred_label AND every alt_label
each get their OWN vector, all pointing back to the same concept_uri. This grows
the index but gives retrieval far more semantic surface area — a resume phrase can
match whichever known phrasing of a skill it is closest to, not just the canonical
name. ``label_type`` on each vector preserves the exact-vs-synonym distinction.

Scope: index construction, persistence, and RAW nearest-neighbor query only. NO
match/no-match decision logic — that is Part 3.3.
"""

from __future__ import annotations

import faiss
import numpy as np

from app.services.rag.taxonomy_schemas import SkillTaxonomyEntry, SkillVectorEntry
from app.services.scoring.embedding_scorer import EmbeddingScorer


class IndexMetadataMismatchError(Exception):
    """Raised when FAISS vector count and metadata length disagree (corruption)."""


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return a float32, L2-normalized copy of ``vectors``.

    We search with IndexFlatIP (inner product). On L2-normalized vectors, inner
    product EQUALS cosine similarity — which is the semantic-closeness measure we
    actually want. Skipping normalization would let longer/higher-magnitude
    embeddings score unfairly high. faiss.normalize_L2 mutates in place, so we
    copy first to avoid surprising the caller.
    """
    out = np.ascontiguousarray(vectors, dtype=np.float32).copy()
    faiss.normalize_L2(out)
    return out


class FAISSSkillIndexBuilder:
    """Builds a FAISS index over ESCO skill labels (+ synonyms)."""

    def __init__(self, embedding_scorer: EmbeddingScorer) -> None:
        self._scorer = embedding_scorer

    def build_index(
        self, taxonomy_entries: list[SkillTaxonomyEntry]
    ) -> tuple[faiss.Index, list[SkillVectorEntry]]:
        """Flatten labels → embed in ONE batch → normalized IndexFlatIP.

        Returns (index, metadata). INVARIANT: metadata[N] describes index vector N;
        any persist/reload must preserve this alignment exactly.
        """
        texts: list[str] = []
        metadata: list[SkillVectorEntry] = []

        for entry in taxonomy_entries:
            flattened = [(entry.preferred_label, "preferred")]
            flattened += [(alt, "alt") for alt in entry.alt_labels]
            for text, label_type in flattened:
                if not text.strip():
                    continue
                metadata.append(
                    SkillVectorEntry(
                        vector_id=len(metadata),
                        concept_uri=entry.concept_uri,
                        matched_text=text,
                        label_type=label_type,  # type: ignore[arg-type]
                    )
                )
                texts.append(text)

        # Single batched embedding call — never a per-text loop at ESCO scale.
        raw = self._scorer.embed_batch(texts) if texts else np.zeros((0, 1))
        vectors = _l2_normalize(np.asarray(raw))

        dim = vectors.shape[1] if vectors.shape[0] > 0 else self._probe_dim()
        index = faiss.IndexFlatIP(dim)
        if vectors.shape[0] > 0:
            index.add(vectors)
        return index, metadata

    def _probe_dim(self) -> int:
        """Embedding dimensionality (for the empty-taxonomy edge case)."""
        return int(np.asarray(self._scorer.embed("dimension probe")).shape[-1])


def save_index(
    index: faiss.Index,
    metadata: list[SkillVectorEntry],
    index_path: str,
    metadata_path: str,
) -> None:
    """Persist the FAISS index (native format) + metadata (JSON lines)."""
    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        for entry in metadata:
            f.write(entry.model_dump_json() + "\n")


def load_index(
    index_path: str, metadata_path: str
) -> tuple[faiss.Index, list[SkillVectorEntry]]:
    """Load index + metadata, validating their alignment.

    Raises IndexMetadataMismatchError if index.ntotal != len(metadata) — a
    mismatch signals silent corruption or version skew, and must fail loudly here
    rather than produce misaligned lookups downstream.
    """
    index = faiss.read_index(index_path)
    metadata: list[SkillVectorEntry] = []
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metadata.append(SkillVectorEntry.model_validate_json(line))

    if index.ntotal != len(metadata):
        raise IndexMetadataMismatchError(
            f"FAISS index has {index.ntotal} vectors but metadata has "
            f"{len(metadata)} entries — index/metadata are out of sync."
        )
    return index, metadata


class FAISSSkillIndexQuerier:
    """Raw nearest-neighbor retrieval over the skill index (no match decisions)."""

    def __init__(
        self,
        index: faiss.Index,
        metadata: list[SkillVectorEntry],
        embedding_scorer: EmbeddingScorer,
    ) -> None:
        if index.ntotal != len(metadata):
            raise IndexMetadataMismatchError(
                f"index.ntotal ({index.ntotal}) != len(metadata) ({len(metadata)})."
            )
        self._index = index
        self._metadata = metadata
        self._scorer = embedding_scorer

    def query_raw(
        self, query_text: str, top_k: int = 5
    ) -> list[tuple[SkillVectorEntry, float]]:
        """Return the top_k nearest (SkillVectorEntry, cosine_similarity) tuples.

        The query is L2-normalized identically to the index vectors, so the
        IndexFlatIP inner-product scores are cosine similarities. Pure retrieval —
        no thresholding or match decisions (that is Part 3.3).
        """
        if self._index.ntotal == 0:
            return []
        query_vec = _l2_normalize(np.asarray(self._scorer.embed(query_text))[None, :])
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, k)

        results: list[tuple[SkillVectorEntry, float]] = []
        for idx, score in zip(indices[0], scores[0], strict=True):
            if idx < 0:  # FAISS pads with -1 when fewer than k results exist.
                continue
            results.append((self._metadata[int(idx)], float(score)))
        return results

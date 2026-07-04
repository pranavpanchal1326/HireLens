"""Sentence-transformer semantic scorer (Phase 2.2).

Produces the ``embedding_score`` field of the locked FeatureVector (Phase 0.2) —
a float in [0.0, 1.0] measuring SEMANTIC similarity between a resume and a JD.

Per PRD §5 this is the deliberately-PRETRAINED layer (all-MiniLM-L6-v2 used
as-is) — the honest, correct engineering choice for semantic similarity at this
scale, and clearly distinct from the TF-IDF layer (trained by us) and the ML
re-ranker (also trained by us). It catches meaning-level matches like
"led a team" ≈ "people management" that lexical TF-IDF structurally cannot.

Standalone by design: meaningful with zero other scoring components active, which
makes pipeline version ``v2-embeddings`` an honest, isolated ablation stage.
"""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# PRD §11 LOCKED model choice: small, free, CPU-fast, no GPU required.
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
# all-MiniLM-L6-v2 accepts up to 256 word-piece tokens; longer input is
# truncated by the model. Long resumes therefore lose their tail — an honest
# limitation to note in the capstone Limitations section (PRD §14 item 9).
_MAX_SEQ_LENGTH = 256
_SCORE_PRECISION = 6


class EmbeddingScorer:
    """Loads all-MiniLM-L6-v2 once and scores semantic resume-vs-JD similarity."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        # Load the model ONCE. Reloading a transformer per call is a severe,
        # common performance mistake (seconds each). Pin device="cpu": free-tier
        # hosting (HF Spaces / Render free) is CPU-only, and assuming GPU would
        # silently break in production.
        self._model = SentenceTransformer(model_name, device="cpu")
        self._model.max_seq_length = _MAX_SEQ_LENGTH
        logger.info("Loaded embedding model %s on CPU", model_name)

    def embed(self, text: str) -> np.ndarray:
        """Encode a single text into its embedding vector.

        Text longer than the model's 256-token limit is truncated by the model
        (truncation, not failure) — see module note on this limitation.
        """
        vector: np.ndarray = self._model.encode(
            text, convert_to_numpy=True, normalize_embeddings=False
        )
        return vector

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Encode many texts at once (efficient; used by 2.3 caching / Phase 3 RAG).

        Produces the same per-text vectors as calling ``embed`` individually.
        """
        vectors: np.ndarray = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=False
        )
        return vectors

    def score(self, resume_text: str, jd_text: str) -> float:
        """Semantic cosine similarity mapped into [0.0, 1.0].

        Raw cosine similarity of sentence embeddings is theoretically in [-1, 1].
        Although negative values are rare for real text, we map the FULL range via
        ``(cos + 1) / 2`` so a negative similarity can never produce an
        out-of-contract value; the final clip is a floating-point safety net.
        Deterministic: these models have no dropout/randomness at inference.
        """
        resume_vec = self.embed(resume_text)
        jd_vec = self.embed(jd_text)
        return self.normalized_similarity(resume_vec, jd_vec)

    @staticmethod
    def normalized_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity of two embedding vectors, mapped/clipped to [0,1].

        Shared entry point so the cache-aware wrapper (2.3) scores identically
        without re-implementing the (cos+1)/2 normalization.
        """
        cosine = EmbeddingScorer._cosine(a, b)
        mapped = (cosine + 1.0) / 2.0
        clipped = min(1.0, max(0.0, mapped))
        return round(clipped, _SCORE_PRECISION)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0  # An all-zero (empty) embedding has no defined direction.
        return float(np.dot(a, b) / denom)

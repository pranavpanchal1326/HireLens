"""RAG Similar-Case Lookup — calibration retrieval (Phase 3.4).

Given a newly-scored resume/JD pair, retrieves the top-k most similar PAST scored
cases so the orchestrator (Phase 4, not built here) can sanity-check whether a new
score is consistent with precedent or an outlier worth flagging.

=========================== UPSTREAM DEPENDENCIES ===========================
Everything below is FIXED by an earlier phase and only imported/called here —
never modified:
  - Embedding function .......... EmbeddingScorer (Phase 2.2/2.3;
                                  sentence-transformers all-MiniLM-L6-v2, CPU).
  - Vector store technology ..... FAISS (Phase 3.2). NOTE: cases need their OWN
                                  FAISS index — skill vectors (384-dim skill
                                  labels) and case vectors (768-dim resume⊕JD)
                                  are different spaces and CANNOT share the 3.2
                                  index instance. Same technology + same model,
                                  separate index. (Design note flagged to human.)
  - Score lineage ............... pipeline_version strings come from ScoreResult
                                  (Phase 0.2) using the Phase 0.3 locked enum.
Locally owned here: SimilarCaseStore, StoredCase, CalibrationResult.
=============================================================================

Scope guardrails: NOT the Ambiguity Flagger (Phase 4.3), NOT the orchestrator
(Phase 4), NOT any API/UI (Phase 7). No embedding retraining or model swap.
"""

from __future__ import annotations

import statistics
import uuid
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
from pydantic import BaseModel, Field

from app.services.scoring.embedding_scorer import EmbeddingScorer

# Provisional deviation (in 0-100 score points) beyond the median of similar
# cases at which a new score is flagged an outlier. PROVISIONAL — pending Phase 5
# ground-truth validation, mirroring the honesty pattern of the other tunables.
OUTLIER_DEVIATION_THRESHOLD = 25.0
_CASE_SCORE_PRECISION = 4


class StoredCase(BaseModel):
    """One persisted past case's metadata (its vector lives in the FAISS index).

    This is an INTERNAL storage structure for this module — NOT a change to the
    Phase 0.2 canonical schemas. Its score/confidence/version fields are copied
    from a finalized ScoreResult (Phase 0.2) when the orchestrator stores a case.
    """

    case_id: str
    final_score: int = Field(ge=0, le=100)
    confidence_level: str  # ScoreResult.ConfidenceLevel value (Phase 0.2)
    pipeline_version: str  # Phase 0.3 locked enum value
    timestamp: str  # ISO 8601 UTC


class CalibrationResult(BaseModel):
    """Fully inspectable calibration signal (no black box — Design Blueprint P1)."""

    is_outlier: bool
    deviation: float
    similar_case_ids: list[str]
    similar_case_scores: list[int]
    low_sample_warning: bool


class SimilarCaseStore:
    """Append-only FAISS store of past scored cases + calibration lookups."""

    def __init__(
        self,
        embedding_scorer: EmbeddingScorer,
        index_path: str | None = None,
        metadata_path: str | None = None,
    ) -> None:
        """embedding_scorer: the Phase 2.2 embedder (reused, never re-instantiated
        with a different model). index_path/metadata_path: optional persistence."""
        self._scorer = embedding_scorer
        self._index_path = index_path
        self._metadata_path = metadata_path
        self._dim = 2 * self._probe_embedding_dim()  # concat of resume ⊕ jd
        self._index: faiss.Index = faiss.IndexFlatIP(self._dim)
        self._metadata: list[StoredCase] = []
        if index_path and metadata_path and Path(index_path).exists():
            self._load()

    # --- embedding ------------------------------------------------------------
    def _probe_embedding_dim(self) -> int:
        return int(np.asarray(self._scorer.embed("probe")).shape[-1])

    def build_case_embedding(self, resume_text: str, jd_text: str) -> np.ndarray:
        """Combined case vector = L2(concat(L2(resume_emb), L2(jd_emb))).

        DESIGN CHOICE (combined, not separate): a similar case = a similar resume
        matched against a similar JD. Independently normalizing each half before
        concatenation keeps both signals present and equally weighted; the final
        L2 makes IndexFlatIP inner product a cosine similarity (consistent with
        Phase 3.2). One vector per case → single clean lookup.
        """
        resume_vec = self._unit(np.asarray(self._scorer.embed(resume_text)))
        jd_vec = self._unit(np.asarray(self._scorer.embed(jd_text)))
        combined = np.concatenate([resume_vec, jd_vec]).astype(np.float32)
        return self._unit(combined)

    @staticmethod
    def _unit(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return vec.astype(np.float32)
        return (vec / norm).astype(np.float32)

    # --- write path -----------------------------------------------------------
    def store_case(self, case: StoredCase, case_embedding: np.ndarray) -> None:
        """Persist a finalized case (append-only) so future lookups include it.

        Called by the Phase 4 orchestrator after a score is finalized (that wiring
        is NOT built here). ``case_embedding`` should come from build_case_embedding.
        """
        vec = self._unit(np.asarray(case_embedding, dtype=np.float32))[None, :]
        self._index.add(vec)
        self._metadata.append(case)
        if self._index_path and self._metadata_path:
            self._save()

    # --- read path ------------------------------------------------------------
    def retrieve_similar_cases(
        self, new_case_embedding: np.ndarray, k: int = 5
    ) -> list[tuple[StoredCase, float]]:
        """Return up to k nearest past (StoredCase, cosine_similarity) tuples.

        Cold start: if fewer than k cases exist, returns whatever is available
        (never errors, never pads with fake cases). The low-sample judgment is made
        in calibration_check.
        """
        if self._index.ntotal == 0:
            return []
        query = self._unit(np.asarray(new_case_embedding, dtype=np.float32))[None, :]
        take = min(k, self._index.ntotal)
        scores, indices = self._index.search(query, take)
        results: list[tuple[StoredCase, float]] = []
        for idx, score in zip(indices[0], scores[0], strict=True):
            if idx < 0:
                continue
            results.append((self._metadata[int(idx)], float(score)))
        return results

    def calibration_check(
        self,
        new_score: int,
        similar_cases: list[tuple[StoredCase, float]],
        requested_k: int = 5,
    ) -> CalibrationResult:
        """Compare new_score to the median score of similar past cases.

        Returns a fully inspectable signal. Cold start (fewer than requested_k
        similar cases) sets low_sample_warning=True so a thin-evidence result is
        never dressed up as confident (Design Blueprint P3). With zero cases, no
        outlier judgment is possible → is_outlier False, deviation 0.0.
        """
        scores = [case.final_score for case, _ in similar_cases]
        low_sample = len(similar_cases) < requested_k
        if not scores:
            return CalibrationResult(
                is_outlier=False,
                deviation=0.0,
                similar_case_ids=[],
                similar_case_scores=[],
                low_sample_warning=True,
            )
        median = statistics.median(scores)
        deviation = round(abs(new_score - median), _CASE_SCORE_PRECISION)
        # Do not assert outlier on thin evidence: only flag when the sample is
        # sufficient AND the deviation is large.
        is_outlier = (not low_sample) and deviation > OUTLIER_DEVIATION_THRESHOLD
        return CalibrationResult(
            is_outlier=is_outlier,
            deviation=deviation,
            similar_case_ids=[case.case_id for case, _ in similar_cases],
            similar_case_scores=scores,
            low_sample_warning=low_sample,
        )

    # --- persistence ----------------------------------------------------------
    def _save(self) -> None:
        assert self._index_path and self._metadata_path
        Path(self._index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, self._index_path)
        with open(self._metadata_path, "w", encoding="utf-8") as f:
            for case in self._metadata:
                f.write(case.model_dump_json() + "\n")

    def _load(self) -> None:
        assert self._index_path and self._metadata_path
        self._index = faiss.read_index(self._index_path)
        self._metadata = []
        with open(self._metadata_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._metadata.append(StoredCase.model_validate_json(line))


def make_case_id() -> str:
    """Convenience id/timestamp helpers for callers that need them."""
    return str(uuid.uuid4())


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()

"""Combined hybrid scorer (Phase 2.4) — the ``v3-hybrid`` pipeline stage.

Merges the lexical TF-IDF score (2.1) and the semantic embedding score (2.3) into
a single weighted score and a full ScoreResult. This is one of the five LOCKED
ablation stages (Phase 0.3 / PRD §7.2).

Honesty rules honored here:
  - Only tfidf_score and embedding_score are genuinely computed at this stage;
    skill_overlap_pct / exp_match / edu_match are set to EXACTLY 0.0 (not
    fabricated), with pipeline_version="v3-hybrid" as the traceable signal of why.
  - Weights are configurable (PRD §3.2 recruiter feature); the default is a
    documented PLACEHOLDER pending Phase 6 grid search against ground truth
    (PRD §8.2) — NOT a tuned result.
  - scoring_confidence here is a simple INTERIM heuristic, explicitly not the
    calibrated model-based confidence Phase 6 will produce.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, model_validator

from app.schemas.parsing import ParsedJobDescription, ParsedResume
from app.schemas.pipeline import PipelineVersion
from app.schemas.scoring import FeatureVector, ScoreResult
from app.services.confidence.confidence_utils import confidence_to_band
from app.services.scoring.embedding_cache import CachedEmbeddingScorer
from app.services.scoring.text_preparation import (
    prepare_jd_text_for_scoring,
    prepare_resume_text_for_scoring,
)
from app.services.scoring.tfidf_scorer import TFIDFScorer

# Interim confidence: only 2 of 5 features are active at v3-hybrid, so the ceiling
# is deliberately modest. Replaced by Phase 6's calibrated scoring_confidence.
_INTERIM_CONFIDENCE_BASELINE = 0.6
_CONFIDENCE_PRECISION = 6


class HybridWeights(BaseModel):
    """Weights for combining tfidf_score and embedding_score. Must sum to 1.0."""

    tfidf_weight: float
    embedding_weight: float

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> HybridWeights:
        total = self.tfidf_weight + self.embedding_weight
        if abs(total - 1.0) >= 1e-6:
            raise ValueError(
                f"tfidf_weight + embedding_weight must equal 1.0, got {total}."
            )
        return self


# PLACEHOLDER default — equal weighting is the most defensible uninformed prior
# until Phase 6's grid search tunes these against real ground-truth data
# (PRD §8.2). This is NOT a final tuned value.
DEFAULT_HYBRID_WEIGHTS = HybridWeights(tfidf_weight=0.5, embedding_weight=0.5)


def _round_half_up_to_100(value_0_1: float) -> int:
    """Scale a [0,1] score to a 0-100 int using round-HALF-UP.

    We deliberately avoid Python's banker's rounding: for a user-facing score,
    predictable "0.5 rounds up" behavior is more intuitive than statistical
    round-half-to-even. UX legibility outranks rounding-convention purity here.
    """
    return int(math.floor(value_0_1 * 100 + 0.5))


class HybridScorer:
    """Combines TF-IDF (2.1) and cached embeddings (2.3) into a ScoreResult."""

    def __init__(
        self,
        tfidf_scorer: TFIDFScorer,
        cached_embedding_scorer: CachedEmbeddingScorer,
        weights: HybridWeights = DEFAULT_HYBRID_WEIGHTS,
    ) -> None:
        self.tfidf_scorer = tfidf_scorer
        self.cached_embedding_scorer = cached_embedding_scorer
        self.weights = weights

    def _raw_scores(
        self,
        resume_id: str,
        parsed_resume: ParsedResume,
        jd_id: str,
        parsed_jd: ParsedJobDescription,
    ) -> tuple[float, float]:
        """Compute the two raw sub-scores ONCE. Returns (tfidf_score, embed_score)."""
        resume_text = prepare_resume_text_for_scoring(parsed_resume)
        jd_text = prepare_jd_text_for_scoring(parsed_jd)
        tfidf_score = self.tfidf_scorer.score(resume_text, jd_text)
        embedding_score = self.cached_embedding_scorer.score(
            resume_id, resume_text, jd_id, jd_text
        )
        return tfidf_score, embedding_score

    def _interim_confidence(self, tfidf_score: float, embedding_score: float) -> float:
        """Interim scoring_confidence heuristic (NOT Phase 6's calibrated value).

        Capped low because only 2/5 features are active, and scaled by how much
        the two available signals AGREE: strong disagreement (one high, one low)
        signals ambiguity → lower confidence. Range ends up in [0.3, 0.6].
        """
        agreement = 1.0 - abs(tfidf_score - embedding_score)  # 1.0 = perfect agree
        confidence = _INTERIM_CONFIDENCE_BASELINE * (0.5 + 0.5 * agreement)
        return round(confidence, _CONFIDENCE_PRECISION)

    def _assemble(
        self,
        resume_id: str,
        parsed_resume: ParsedResume,
        jd_id: str,
        tfidf_score: float,
        embedding_score: float,
        weights: HybridWeights,
        pipeline_version: PipelineVersion,
    ) -> ScoreResult:
        """Build a ScoreResult from already-computed raw scores + weights."""
        feature_vector = FeatureVector(
            tfidf_score=tfidf_score,
            embedding_score=embedding_score,
            # Not yet computed at v3-hybrid stage — honest 0.0, see pipeline_version.
            skill_overlap_pct=0.0,
            exp_match=0.0,
            edu_match=0.0,
        )
        weighted = (
            tfidf_score * weights.tfidf_weight
            + embedding_score * weights.embedding_weight
        )
        final_score = _round_half_up_to_100(weighted)
        confidence = self._interim_confidence(tfidf_score, embedding_score)

        return ScoreResult(
            resume_id=resume_id,
            jd_id=jd_id,
            final_score=final_score,
            feature_vector=feature_vector,
            scoring_confidence=confidence,
            confidence_level=confidence_to_band(confidence),
            # Carried through from parsing so a bad score is traceable to
            # "couldn't read it" vs "genuine mismatch" (PRD §8.2).
            parsing_confidence=parsed_resume.parsing_confidence,
            pipeline_version=pipeline_version.value,
        )

    def compute_hybrid_score(
        self,
        resume_id: str,
        parsed_resume: ParsedResume,
        jd_id: str,
        parsed_jd: ParsedJobDescription,
        weights: HybridWeights | None = None,
    ) -> ScoreResult:
        """Full v3-hybrid ScoreResult. ``weights`` overrides the instance default
        (per-request configurable weights, PRD §3.2)."""
        effective = weights if weights is not None else self.weights
        tfidf_score, embedding_score = self._raw_scores(
            resume_id, parsed_resume, jd_id, parsed_jd
        )
        return self._assemble(
            resume_id,
            parsed_resume,
            jd_id,
            tfidf_score,
            embedding_score,
            effective,
            PipelineVersion.V3_HYBRID,
        )

    def compute_ablation_stage_scores(
        self,
        resume_id: str,
        parsed_resume: ParsedResume,
        jd_id: str,
        parsed_jd: ParsedJobDescription,
    ) -> dict[str, ScoreResult]:
        """Produce v1-tfidf, v2-embeddings, and v3-hybrid ScoreResults in one call.

        The raw tfidf/embedding scores are computed EXACTLY ONCE and reused across
        all three weight combinations — Phase 5.3's ablation study runs this over
        many pairs, so recomputing would be wasteful.
        """
        tfidf_score, embedding_score = self._raw_scores(
            resume_id, parsed_resume, jd_id, parsed_jd
        )
        stages = {
            PipelineVersion.V1_TFIDF: HybridWeights(
                tfidf_weight=1.0, embedding_weight=0.0
            ),
            PipelineVersion.V2_EMBEDDINGS: HybridWeights(
                tfidf_weight=0.0, embedding_weight=1.0
            ),
            PipelineVersion.V3_HYBRID: DEFAULT_HYBRID_WEIGHTS,
        }
        return {
            version.value: self._assemble(
                resume_id,
                parsed_resume,
                jd_id,
                tfidf_score,
                embedding_score,
                weights,
                version,
            )
            for version, weights in stages.items()
        }

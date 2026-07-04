"""Feature Engineering Pipeline (Phase 6.1).

Defines the FeatureVector schema and implements extraction and normalization
functions for the canonical 5-dimensional feature vector.

LOCKED CONTRACT — Maps to: PRD §5 (feature vector) and Design Blueprint §6.2.
Field ORDER and NAMING:
[tfidf_score, embedding_score, skill_overlap_pct, exp_match, edu_match]
Each value is normalized to [0.0, 1.0].
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.schemas.parsing import ParsedJobDescription, ParsedResume
from app.schemas.scoring import FeatureVector
from app.services.scoring.text_preparation import (
    prepare_jd_text_for_scoring,
    prepare_resume_text_for_scoring,
)

if TYPE_CHECKING:
    from app.services.orchestration.agent_orchestrator import OrchestratorTools
    from app.services.rag.skill_matcher import SkillMatcher
    from app.services.rag.taxonomy_schemas import SkillTaxonomyEntry
    from app.services.scoring.embedding_cache import CachedEmbeddingScorer
    from app.services.scoring.experience_matcher import ExperienceMatcher
    from app.services.scoring.tfidf_scorer import TFIDFScorer

logger = logging.getLogger(__name__)

# ============================ EDUCATION HIERARCHY ============================
# Ordered ranks for education levels extracted by the parsing/structuring layers.
# Crucial for matching JD requirements against candidate achievements.
EDUCATION_LEVEL_HIERARCHY = {
    "None": 0,
    "Associate's": 1,
    "Bachelor's": 2,
    "Master's": 3,
    "PhD": 4,
}

# Explicit lookup table mapping underqualification to a match score.
# Keys are the rank of required degree; values map candidate degree rank to score.
UNDERQUALIFICATION_RULE_TABLE = {
    4: {3: 0.75, 2: 0.50, 1: 0.25, 0: 0.00},  # PhD required
    3: {2: 0.67, 1: 0.33, 0: 0.00},  # Master's required
    2: {1: 0.50, 0: 0.00},  # Bachelor's required
    1: {0: 0.00},  # Associate's required
}


# ============================ CUSTOM EXCEPTIONS =============================
class FeatureEngineeringError(Exception):
    """Base exception for all feature engineering pipeline failures."""


class UpstreamToolError(FeatureEngineeringError):
    """Raised when an upstream scoring or matching tool fails during extraction."""

    def __init__(
        self, feature_name: str, message: str, original_exception: Exception
    ) -> None:
        self.feature_name = feature_name
        self.original_exception = original_exception
        msg = (
            f"Feature '{feature_name}' extraction failed due to "
            f"upstream tool error: {message}"
        )
        super().__init__(msg)


class FeatureValidationError(FeatureEngineeringError):
    """Raised when a feature value falls outside the canonical [0.0, 1.0] bounds."""


# ============================ EXTRACTION LOGIC ==============================


def extract_tfidf_score(
    resume: ParsedResume, jd: ParsedJobDescription, tfidf_scorer: TFIDFScorer
) -> float:
    """Extract and normalize the TF-IDF lexical match score.

    Upstream Called:
        Phase 2.1 TFIDFScorer.score()

    Raw Output Range:
        The TFIDFScorer.score() method calculates the cosine similarity of the
        two TF-IDF vectors, which mathematically falls within [0.0, 1.0].
        The scorer itself already performs bounds clipping as a safety net.

    Normalization:
        No scaling is needed as the cosine similarity of non-negative TF-IDF
        vectors is naturally bounded in [0.0, 1.0]. We strictly verify the
        output is within bounds.
    """
    try:
        resume_text = prepare_resume_text_for_scoring(resume)
        jd_text = prepare_jd_text_for_scoring(jd)
        score = tfidf_scorer.score(resume_text, jd_text)
    except Exception as e:
        raise UpstreamToolError("tfidf_score", str(e), e) from e

    if not (0.0 <= score <= 1.0):
        raise FeatureValidationError(
            f"tfidf_score {score} is out of bounds [0.0, 1.0]."
        )
    return score


def extract_embedding_score(
    resume: ParsedResume,
    jd: ParsedJobDescription,
    cached_embedding_scorer: CachedEmbeddingScorer,
) -> float:
    """Extract and normalize the semantic embedding similarity score.

    Upstream Called:
        Phase 2.2/2.3 CachedEmbeddingScorer.score()

    Raw Output Range:
        Raw cosine similarity of SentenceTransformer embeddings is in [-1.0, 1.0].

    Normalization:
        The cached embedding scorer already normalizes this range using the
        transformation (cosine + 1.0) / 2.0, yielding a float in [0.0, 1.0].
        This ensures that negative similarity cannot break downstream rendering.
        We verify and enforce the [0.0, 1.0] bounds.
    """
    try:
        resume_text = prepare_resume_text_for_scoring(resume)
        jd_text = prepare_jd_text_for_scoring(jd)
        score = cached_embedding_scorer.score(
            resume.document_id, resume_text, jd.document_id, jd_text
        )
    except Exception as e:
        raise UpstreamToolError("embedding_score", str(e), e) from e

    if not (0.0 <= score <= 1.0):
        raise FeatureValidationError(
            f"embedding_score {score} is out of bounds [0.0, 1.0]."
        )
    return score


def extract_skill_overlap_pct(
    resume: ParsedResume,
    jd: ParsedJobDescription,
    skill_matcher: SkillMatcher,
    taxonomy_entries: list[SkillTaxonomyEntry],
) -> float:
    """Extract and normalize the RAG skill overlap metric.

    Upstream Called:
        Phase 3.3 SkillMatcher.match_resume_to_jd()

    Raw Output Range:
        Calculates the weighted ratio of matched required and preferred skills,
        producing a float in [0.0, 1.0].

    NAMING VS SCALE CLARIFICATION:
        The PRD §5 names this feature 'skill_overlap%' (percentage), but the
        canonical FeatureVector contract requires a normalized float in the
        range [0.0, 1.0] (e.g. 0.75 instead of 75.0) to maintain consistency
        with the other features and satisfy the downstream glyph rendering formula:
        tip radius = 34 + value * 40. We explicitly enforce the [0.0, 1.0] range.
    """
    try:
        overlap, _, _ = skill_matcher.match_resume_to_jd(
            resume.skills,
            jd.required_skills,
            jd.preferred_skills,
            taxonomy_entries,
        )
    except Exception as e:
        raise UpstreamToolError("skill_overlap_pct", str(e), e) from e

    if not (0.0 <= overlap <= 1.0):
        raise FeatureValidationError(
            f"skill_overlap_pct {overlap} is out of bounds [0.0, 1.0]."
        )
    return overlap


def extract_exp_match(
    resume: ParsedResume,
    jd: ParsedJobDescription,
    experience_matcher: ExperienceMatcher,
) -> float:
    """Extract and normalize the experience/years match score.

    Upstream Called:
        Phase 4.4 ExperienceMatcher.match()

    EXP_MATCH SCORING CURVE JUSTIFICATION:
        Exceeding the required years is capped at 1.0. This decision reflects the
        fact that once a candidate meets the minimum required years of experience,
        additional years represent diminishing returns for the baseline requirements
        of the role. Under-qualification scales linearly in [0.0, 1.0] based on the
        proportion of required years met (i.e., actual_years / required_years).
        This capped linear model is the most stable and interpretable representation
        of experience match strength.
    """
    try:
        score = experience_matcher.match(resume, jd)
    except Exception as e:
        raise UpstreamToolError("exp_match", str(e), e) from e

    if not (0.0 <= score <= 1.0):
        raise FeatureValidationError(f"exp_match {score} is out of bounds [0.0, 1.0].")
    return score


def extract_edu_match(resume: ParsedResume, jd: ParsedJobDescription) -> float:
    """Extract and normalize the education level match score.

    EDU_MATCH SCORING CURVE JUSTIFICATION:
        Meeting or exceeding the required education level yields a perfect 1.0.
        Exceeding is capped at 1.0 to avoid over-weighting candidates with advanced
        degrees beyond the JD criteria. Under-qualification is scored deterministically
        based on an explicit lookup table that penalizes the rank difference
        (e.g., Bachelor's for a Master's requirement receives 0.67).
    """
    try:
        candidate_highest = get_candidate_highest_education(resume)
        required = jd.required_education_level
        score = match_education(candidate_highest, required)
    except Exception as e:
        raise UpstreamToolError("edu_match", str(e), e) from e

    if not (0.0 <= score <= 1.0):
        raise FeatureValidationError(f"edu_match {score} is out of bounds [0.0, 1.0].")
    return score


# ============================ HELPERS =======================================


def get_candidate_highest_education(resume: ParsedResume) -> str:
    """Determine the highest education level from ParsedResume's education entries.

    Returns one of the canonical keys in EDUCATION_LEVEL_HIERARCHY.
    If no education entries exist or no known degree is found, returns 'None'.
    """
    if not resume.education:
        return "None"

    highest_rank = 0
    highest_level = "None"

    for entry in resume.education:
        if not entry.degree:
            continue
        deg = entry.degree.strip()
        # Direct lookup first
        rank = EDUCATION_LEVEL_HIERARCHY.get(deg, 0)

        # Defensive fallback keyword matching if the string was not fully canonicalized
        if rank == 0:
            deg_lower = deg.lower()
            if any(kw in deg_lower for kw in ("ph.d", "phd", "doctorate", "doctoral")):
                rank = EDUCATION_LEVEL_HIERARCHY["PhD"]
                deg = "PhD"
            elif any(
                kw in deg_lower
                for kw in ("master", "m.s", "msc", "m.sc", "m.a", "mba", "m.eng")
            ):
                rank = EDUCATION_LEVEL_HIERARCHY["Master's"]
                deg = "Master's"
            elif any(
                kw in deg_lower
                for kw in ("bachelor", "b.s", "bsc", "b.sc", "b.a", "b.eng", "b.tech")
            ):
                rank = EDUCATION_LEVEL_HIERARCHY["Bachelor's"]
                deg = "Bachelor's"
            elif any(kw in deg_lower for kw in ("associate", "a.a", "a.s")):
                rank = EDUCATION_LEVEL_HIERARCHY["Associate's"]
                deg = "Associate's"

        if rank > highest_rank:
            highest_rank = rank
            highest_level = deg

    return highest_level


def match_education(candidate_highest: str, required: str | None) -> float:
    """Compare candidate's highest education rank against JD's required level rank.

    If JD has no requirement or required is unknown, returns 1.0.
    If candidate meets or exceeds requirement, returns 1.0.
    If candidate is underqualified, returns the explicit score from
    UNDERQUALIFICATION_RULE_TABLE.
    """
    if required is None or required not in EDUCATION_LEVEL_HIERARCHY:
        return 1.0

    required_rank = EDUCATION_LEVEL_HIERARCHY[required]
    candidate_rank = EDUCATION_LEVEL_HIERARCHY.get(candidate_highest, 0)

    if candidate_rank >= required_rank:
        return 1.0

    sub_table = UNDERQUALIFICATION_RULE_TABLE.get(required_rank)
    if sub_table is None:
        return 0.0
    return sub_table.get(candidate_rank, 0.0)


# ============================ UNIFIED ENTRY POINT ===========================


def extract_feature_vector(
    parsed_resume: ParsedResume,
    parsed_jd: ParsedJobDescription,
    tools: OrchestratorTools,
) -> FeatureVector:
    """Extract and validate the canonical 5-dimensional feature vector.

    Named-exception handling per feature is used so failures in individual
    upstream tools are attributed directly to the specific feature.
    """
    if tools is None:
        raise FeatureEngineeringError(
            "extract_feature_vector requires `tools` parameter."
        )

    # 1. tfidf_score
    tfidf_score = extract_tfidf_score(
        parsed_resume, parsed_jd, tools.hybrid_scorer.tfidf_scorer
    )

    # 2. embedding_score
    embedding_score = extract_embedding_score(
        parsed_resume, parsed_jd, tools.hybrid_scorer.cached_embedding_scorer
    )

    # 3. skill_overlap_pct
    skill_overlap_pct = extract_skill_overlap_pct(
        parsed_resume, parsed_jd, tools.skill_matcher, tools.taxonomy_entries
    )

    # 4. exp_match
    exp_match = extract_exp_match(parsed_resume, parsed_jd, tools.experience_matcher)

    # 5. edu_match
    edu_match = extract_edu_match(parsed_resume, parsed_jd)

    vector = FeatureVector(
        tfidf_score=tfidf_score,
        embedding_score=embedding_score,
        skill_overlap_pct=skill_overlap_pct,
        exp_match=exp_match,
        edu_match=edu_match,
    )

    # Explicit audit of the final schema object before releasing it downstream.
    # Pydantic validates this on instantiation, but this ensures a clean,
    # auditable trace.
    for name, value in vector.model_dump().items():
        if not (0.0 <= value <= 1.0):
            raise FeatureValidationError(
                f"Feature '{name}' has value {value} which is outside the "
                "strict [0.0, 1.0] normalized bounds."
            )

    return vector

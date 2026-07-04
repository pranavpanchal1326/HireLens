"""Unit tests for the Feature Engineering Pipeline (Phase 6.1)."""

from __future__ import annotations

import pytest

from app.schemas.parsing import EducationEntry, ParsedJobDescription, ParsedResume
from app.schemas.scoring import FeatureVector
from app.services.orchestration.agent_orchestrator import OrchestratorTools
from app.services.scoring.feature_engineering import (
    FeatureValidationError,
    UpstreamToolError,
    extract_edu_match,
    extract_embedding_score,
    extract_exp_match,
    extract_feature_vector,
    extract_skill_overlap_pct,
    extract_tfidf_score,
    get_candidate_highest_education,
    match_education,
)


def test_feature_contract_order_invariant_across_phase6() -> None:
    """Regression (Phase 6.X audit, Pass 1): the 5-feature name/order contract must
    stay byte-identical across the FeatureVector schema, 6.3's FEATURE_ORDER, and
    6.2's array build; and 6.4's ENSEMBLE_KEYS must be exactly the first 4 (edu_match
    intentionally excluded). Turns the manual audit trace into an executable guard."""
    from app.services.scoring.feature_importance import FEATURE_ORDER
    from app.services.scoring.grid_search_tuning import ENSEMBLE_KEYS

    canonical = [
        "tfidf_score",
        "embedding_score",
        "skill_overlap_pct",
        "exp_match",
        "edu_match",
    ]
    # Contract schema field order (Pydantic preserves declaration order).
    assert list(FeatureVector.model_fields.keys()) == canonical
    # Phase 6.3 importance key order.
    assert FEATURE_ORDER == canonical
    # Phase 6.4 ensemble = first 4 features, edu_match excluded by design.
    assert ENSEMBLE_KEYS == canonical[:4]
    assert "edu_match" not in ENSEMBLE_KEYS


# ============================ STUB SCORERS & MATCHERS ========================


class StubTFIDFScorer:
    def __init__(self, score_value: float | Exception) -> None:
        self.score_value = score_value

    def score(self, resume_text: str, jd_text: str) -> float:
        if isinstance(self.score_value, Exception):
            raise self.score_value
        return self.score_value


class StubCachedEmbeddingScorer:
    def __init__(self, score_value: float | Exception) -> None:
        self.score_value = score_value

    def score(
        self, resume_id: str, resume_text: str, jd_id: str, jd_text: str
    ) -> float:
        if isinstance(self.score_value, Exception):
            raise self.score_value
        return self.score_value


class StubSkillMatcher:
    def __init__(self, return_val: tuple[float, list, list] | Exception) -> None:
        self.return_val = return_val

    def match_resume_to_jd(self, *args, **kwargs) -> tuple[float, list, list]:
        if isinstance(self.return_val, Exception):
            raise self.return_val
        return self.return_val


class StubExperienceMatcher:
    def __init__(self, score_value: float | Exception) -> None:
        self.score_value = score_value

    def match(self, resume: ParsedResume, jd: ParsedJobDescription) -> float:
        if isinstance(self.score_value, Exception):
            raise self.score_value
        return self.score_value


class StubHybridScorer:
    def __init__(
        self, tfidf: StubTFIDFScorer, embedding: StubCachedEmbeddingScorer
    ) -> None:
        self.tfidf_scorer = tfidf
        self.cached_embedding_scorer = embedding


# ============================ HELPERS =======================================


def _resume(
    skills: list[str], education: list[EducationEntry], years: float | None = 2.0
) -> ParsedResume:
    return ParsedResume(
        raw_text="Test resume text",
        skills=skills,
        experience=[],
        education=education,
        total_years_experience=years,
        contact_info_present=False,
        parsing_confidence=1.0,
        pipeline_version="parser-v1",
    )


def _jd(
    skills: list[str], years: float | None = 3.0, edu_level: str | None = None
) -> ParsedJobDescription:
    return ParsedJobDescription(
        raw_text="Test JD text",
        required_skills=skills,
        preferred_skills=[],
        required_years_experience=years,
        required_education_level=edu_level,
        parsing_confidence=1.0,
        pipeline_version="parser-v1",
    )


# ============================ TEST CASES =====================================


# 1. TF-IDF scoring tests
def test_tfidf_extraction_normalization() -> None:
    scorer = StubTFIDFScorer(0.85)
    r = _resume([], [])
    j = _jd([])
    val = extract_tfidf_score(r, j, scorer)  # type: ignore[arg-type]
    assert val == 0.85


def test_tfidf_out_of_bounds_raises() -> None:
    scorer = StubTFIDFScorer(1.2)  # invalid (> 1.0)
    r = _resume([], [])
    j = _jd([])
    with pytest.raises(FeatureValidationError) as exc:
        extract_tfidf_score(r, j, scorer)  # type: ignore[arg-type]
    assert "out of bounds" in str(exc.value)


def test_tfidf_upstream_error_raises() -> None:
    scorer = StubTFIDFScorer(ValueError("TF-IDF backend exploded"))
    r = _resume([], [])
    j = _jd([])
    with pytest.raises(UpstreamToolError) as exc:
        extract_tfidf_score(r, j, scorer)  # type: ignore[arg-type]
    assert "tfidf_score" in exc.value.feature_name
    assert "TF-IDF backend exploded" in str(exc.value)


# 2. Embedding scoring tests
def test_embedding_extraction_normalization() -> None:
    scorer = StubCachedEmbeddingScorer(0.92)
    r = _resume([], [])
    j = _jd([])
    val = extract_embedding_score(r, j, scorer)  # type: ignore[arg-type]
    assert val == 0.92


def test_embedding_out_of_bounds_raises() -> None:
    scorer = StubCachedEmbeddingScorer(-0.1)  # invalid (< 0.0)
    r = _resume([], [])
    j = _jd([])
    with pytest.raises(FeatureValidationError) as exc:
        extract_embedding_score(r, j, scorer)  # type: ignore[arg-type]
    assert "out of bounds" in str(exc.value)


def test_embedding_upstream_error_raises() -> None:
    scorer = StubCachedEmbeddingScorer(RuntimeError("FAISS server down"))
    r = _resume([], [])
    j = _jd([])
    with pytest.raises(UpstreamToolError) as exc:
        extract_embedding_score(r, j, scorer)  # type: ignore[arg-type]
    assert "embedding_score" in exc.value.feature_name
    assert "FAISS server down" in str(exc.value)


# 3. Skill overlap tests
def test_skill_overlap_extraction() -> None:
    matcher = StubSkillMatcher((0.75, [], []))
    r = _resume(["Python"], [])
    j = _jd(["Python"])
    val = extract_skill_overlap_pct(r, j, matcher, [])  # type: ignore[arg-type]
    assert val == 0.75


def test_skill_overlap_out_of_bounds_raises() -> None:
    matcher = StubSkillMatcher((1.5, [], []))  # invalid (> 1.0)
    r = _resume(["Python"], [])
    j = _jd(["Python"])
    with pytest.raises(FeatureValidationError) as exc:
        extract_skill_overlap_pct(r, j, matcher, [])  # type: ignore[arg-type]
    assert "out of bounds" in str(exc.value)


def test_skill_overlap_upstream_error_raises() -> None:
    matcher = StubSkillMatcher(ValueError("Skill lookup failed"))
    r = _resume(["Python"], [])
    j = _jd(["Python"])
    with pytest.raises(UpstreamToolError) as exc:
        extract_skill_overlap_pct(r, j, matcher, [])  # type: ignore[arg-type]
    assert "skill_overlap_pct" in exc.value.feature_name
    assert "Skill lookup failed" in str(exc.value)


# 4. Experience match tests
def test_exp_match_extraction() -> None:
    matcher = StubExperienceMatcher(0.66)
    r = _resume([], [], years=2.0)
    j = _jd([], years=3.0)
    val = extract_exp_match(r, j, matcher)  # type: ignore[arg-type]
    assert val == 0.66


def test_exp_match_out_of_bounds_raises() -> None:
    matcher = StubExperienceMatcher(-0.5)  # invalid (< 0.0)
    r = _resume([], [], years=2.0)
    j = _jd([], years=3.0)
    with pytest.raises(FeatureValidationError) as exc:
        extract_exp_match(r, j, matcher)  # type: ignore[arg-type]
    assert "out of bounds" in str(exc.value)


def test_exp_match_upstream_error_raises() -> None:
    matcher = StubExperienceMatcher(RuntimeError("Exp match error"))
    r = _resume([], [], years=2.0)
    j = _jd([], years=3.0)
    with pytest.raises(UpstreamToolError) as exc:
        extract_exp_match(r, j, matcher)  # type: ignore[arg-type]
    assert "exp_match" in exc.value.feature_name
    assert "Exp match error" in str(exc.value)


# 5. Education matching tests
def test_edu_match_logic_cases() -> None:
    # Rule table checks:
    # 1. No requirement -> 1.0
    assert match_education("Associate's", None) == 1.0
    assert match_education("None", "UnknownLevel") == 1.0

    # 2. Exact match or higher -> 1.0
    assert match_education("Bachelor's", "Bachelor's") == 1.0
    assert match_education("PhD", "Bachelor's") == 1.0
    assert match_education("Master's", "Bachelor's") == 1.0

    # 3. Underqualification - PhD required
    assert match_education("Master's", "PhD") == 0.75
    assert match_education("Bachelor's", "PhD") == 0.50
    assert match_education("Associate's", "PhD") == 0.25
    assert match_education("None", "PhD") == 0.00

    # 4. Underqualification - Master's required
    assert match_education("Bachelor's", "Master's") == 0.67
    assert match_education("Associate's", "Master's") == 0.33
    assert match_education("None", "Master's") == 0.00

    # 5. Underqualification - Bachelor's required
    assert match_education("Associate's", "Bachelor's") == 0.50
    assert match_education("None", "Bachelor's") == 0.00

    # 6. Underqualification - Associate's required
    assert match_education("None", "Associate's") == 0.00


def test_highest_education_fallback_detection() -> None:
    # Test fallback classification on non-canonical degree strings
    r_empty = _resume([], [])
    assert get_candidate_highest_education(r_empty) == "None"

    r_phd = _resume([], [EducationEntry(degree="Doctor of Philosophy (Ph.D.)")])
    assert get_candidate_highest_education(r_phd) == "PhD"

    r_master = _resume(
        [],
        [EducationEntry(degree="MBA in Finance"), EducationEntry(degree="Bachelor's")],
    )
    assert get_candidate_highest_education(r_master) == "Master's"

    r_bachelor = _resume([], [EducationEntry(degree="B.Tech in Computer Science")])
    assert get_candidate_highest_education(r_bachelor) == "Bachelor's"

    r_associate = _resume([], [EducationEntry(degree="A.A. Liberal Arts")])
    assert get_candidate_highest_education(r_associate) == "Associate's"


def test_edu_match_extraction() -> None:
    r = _resume([], [EducationEntry(degree="Master's")])
    j = _jd([], edu_level="PhD")
    score = extract_edu_match(r, j)
    assert score == 0.75


# 6. End-to-end feature vector extraction tests
def test_extract_feature_vector_end_to_end() -> None:
    tools = OrchestratorTools(
        hybrid_scorer=StubHybridScorer(
            StubTFIDFScorer(0.80), StubCachedEmbeddingScorer(0.90)
        ),  # type: ignore[arg-type]
        skill_matcher=StubSkillMatcher((0.70, [], [])),  # type: ignore[arg-type]
        taxonomy_entries=[],
        case_store=None,  # type: ignore[arg-type]
        experience_matcher=StubExperienceMatcher(0.60),  # type: ignore[arg-type]
    )

    r = _resume(["Python"], [EducationEntry(degree="Bachelor's")])
    j = _jd(["Python"], edu_level="Master's")

    vector = extract_feature_vector(r, j, tools)

    assert isinstance(vector, FeatureVector)
    assert vector.tfidf_score == 0.80
    assert vector.embedding_score == 0.90
    assert vector.skill_overlap_pct == 0.70
    assert vector.exp_match == 0.60
    assert vector.edu_match == 0.67


def test_extract_feature_vector_bounds_enforced() -> None:
    # Deliberate out-of-bounds returned by one stub
    tools = OrchestratorTools(
        hybrid_scorer=StubHybridScorer(
            StubTFIDFScorer(1.05), StubCachedEmbeddingScorer(0.90)
        ),  # type: ignore[arg-type]
        skill_matcher=StubSkillMatcher((0.70, [], [])),  # type: ignore[arg-type]
        taxonomy_entries=[],
        case_store=None,  # type: ignore[arg-type]
        experience_matcher=StubExperienceMatcher(0.60),  # type: ignore[arg-type]
    )

    r = _resume(["Python"], [EducationEntry(degree="Bachelor's")])
    j = _jd(["Python"], edu_level="Master's")

    with pytest.raises(FeatureValidationError) as exc:
        extract_feature_vector(r, j, tools)
    assert "out of bounds" in str(exc.value)

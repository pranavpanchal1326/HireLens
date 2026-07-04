"""Tests for the Phase 3.3 semantic skill-matching decision logic."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.services.rag.faiss_index_builder import FAISSSkillIndexBuilder
from app.services.rag.skill_matcher import (
    SEMANTIC_MATCH_THRESHOLD,
    SkillMatcher,
    generate_action_phrased_gap,
)
from app.services.rag.taxonomy_schemas import SkillTaxonomyEntry
from app.services.scoring.embedding_scorer import EmbeddingScorer

_SCORER = EmbeddingScorer()

TAXONOMY = [
    SkillTaxonomyEntry(
        concept_uri="esco/1",
        preferred_label="people management",
        alt_labels=["team leadership", "managing staff"],
    ),
    SkillTaxonomyEntry(
        concept_uri="esco/2",
        preferred_label="python programming",
        alt_labels=["python", "python development"],
    ),
    SkillTaxonomyEntry(
        concept_uri="esco/3",
        preferred_label="financial accounting",
        alt_labels=["bookkeeping"],
    ),
    SkillTaxonomyEntry(
        concept_uri="esco/4",
        preferred_label="graphic design",
        alt_labels=["visual design"],
    ),
]


def _real_matcher() -> SkillMatcher:
    index, metadata = FAISSSkillIndexBuilder(_SCORER).build_index(TAXONOMY)
    from app.services.rag.faiss_index_builder import FAISSSkillIndexQuerier

    return SkillMatcher(FAISSSkillIndexQuerier(index, metadata, _SCORER))


def test_exact_match_varied_casing() -> None:
    m = _real_matcher()
    result = m.match_single_skill(
        "  PYTHON   Programming ", "python programming", TAXONOMY
    )
    assert result.match_type == "exact"
    assert result.similarity_score == 1.0


def test_synonym_match_resolves_as_exact() -> None:
    """resume=alt_label, JD=preferred_label of the SAME concept → precise (exact)."""
    m = _real_matcher()
    result = m.match_single_skill("team leadership", "people management", TAXONOMY)
    assert result.match_type == "exact"
    assert result.matched_via_concept_uri == "esco/1"


def test_semantic_match_no_exact_overlap() -> None:
    m = _real_matcher()
    result = m.match_single_skill(
        "led a team of engineers", "people management", TAXONOMY
    )
    assert result.match_type == "semantic"
    assert result.similarity_score >= SEMANTIC_MATCH_THRESHOLD
    assert result.matched_via_concept_uri == "esco/1"


def test_no_match_unrelated() -> None:
    m = _real_matcher()
    result = m.match_single_skill("financial accounting", "graphic design", TAXONOMY)
    assert result.match_type == "none"


def test_exact_match_short_circuits_faiss() -> None:
    """Exact match must NOT trigger a FAISS query (proves ordering short-circuit)."""
    spy_querier = MagicMock()
    matcher = SkillMatcher(spy_querier)
    result = matcher.match_single_skill("Python", "python", TAXONOMY)
    assert result.match_type == "exact"
    spy_querier.query_raw.assert_not_called()


def test_synonym_match_short_circuits_faiss() -> None:
    spy_querier = MagicMock()
    matcher = SkillMatcher(spy_querier)
    result = matcher.match_single_skill(
        "team leadership", "people management", TAXONOMY
    )
    assert result.match_type == "exact"
    spy_querier.query_raw.assert_not_called()


def test_skill_overlap_pct_weighted_formula() -> None:
    """2 required (1 matched, 1 missing) + 2 preferred (1 matched, 1 missing).
    total=2*1.0 + 2*0.5 = 3.0; matched=1.0 + 0.5 = 1.5; overlap = 0.5."""
    m = _real_matcher()
    resume = ["python programming", "graphic design"]
    required = ["python programming", "people management"]  # 1 match, 1 gap
    preferred = ["graphic design", "financial accounting"]  # 1 match, 1 gap
    overlap, matched, gaps = m.match_resume_to_jd(resume, required, preferred, TAXONOMY)
    assert overlap == 0.5
    assert len(matched) == 2
    assert {g.missing_skill for g in gaps} == {
        "people management",
        "financial accounting",
    }


def test_no_double_counting() -> None:
    """One resume skill cannot be claimed by two JD requirements."""
    m = _real_matcher()
    resume = ["python programming"]  # only ONE resume skill
    required = ["python programming", "python development"]  # both relate to esco/2
    overlap, matched, gaps = m.match_resume_to_jd(resume, required, [], TAXONOMY)
    # The single resume skill is claimed once; the second requirement is a gap.
    assert len(matched) == 1
    assert len(gaps) == 1


def test_gap_language_denylist() -> None:
    """Design Blueprint §12: gap suggestions never use deficiency framing."""
    denylist = ["lack", "missing", "weak", "deficien", "fail", "poor", "insufficient"]
    for skill in ["Python", "team leadership", "financial accounting", "Kubernetes"]:
        text = generate_action_phrased_gap(skill).lower()
        for banned in denylist:
            assert banned not in text, f"'{banned}' appeared in gap text: {text!r}"


def test_gap_generation_is_deterministic() -> None:
    a = generate_action_phrased_gap("Python")
    b = generate_action_phrased_gap("Python")
    assert a == b

"""Tests for the Phase 2.1 TF-IDF lexical scorer.

Uses a small synthetic corpus — unit tests must NOT depend on the real 120k
Kaggle dataset (that's the CLI script's job).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.schemas.parsing import (
    EducationEntry,
    ExperienceEntry,
    ParsedJobDescription,
    ParsedResume,
)
from app.services.scoring.tfidf_scorer import (
    TFIDFScorer,
    prepare_jd_text_for_scoring,
    prepare_resume_text_for_scoring,
)

# 16 short synthetic documents spanning software, data, and unrelated domains so
# IDF has a real (if tiny) vocabulary space to work with.
SYNTHETIC_CORPUS = [
    "python developer with experience in django and flask web frameworks",
    "senior python engineer building rest apis with fastapi and postgresql",
    "java backend engineer spring boot microservices and kubernetes",
    "machine learning engineer tensorflow pytorch and scikit learn models",
    "data scientist python pandas numpy statistics and data visualization",
    "frontend developer react redux typescript and css responsive design",
    "devops engineer aws docker terraform and ci cd pipelines jenkins",
    "sql database administrator postgresql mysql performance tuning and backups",
    "product manager roadmap prioritization stakeholder management and agile",
    "graphic designer adobe photoshop illustrator branding and typography",
    "chef restaurant kitchen management menu planning and food safety",
    "nurse patient care hospital clinical experience and medication administration",
    "accountant financial reporting budgeting tax preparation and auditing",
    "marketing specialist seo content strategy google analytics and campaigns",
    "python data engineer airflow spark etl pipelines and snowflake warehouse",
    "software engineer python java docker and machine learning deployment",
]


@pytest.fixture
def scorer(tmp_path: Path) -> TFIDFScorer:
    s = TFIDFScorer(model_path=tmp_path / "vec.joblib")
    s.fit(SYNTHETIC_CORPUS)
    return s


def test_similar_texts_score_high(scorer: TFIDFScorer) -> None:
    a = "python developer experienced in django and flask web frameworks"
    b = "python developer with django flask web framework experience"
    score = scorer.score(a, b)
    # Heavy lexical overlap → high similarity. 0.5 is a conservative floor given
    # the tiny corpus dampens IDF weighting.
    assert score >= 0.5


def test_unrelated_texts_score_low(scorer: TFIDFScorer) -> None:
    software = "senior python engineer building rest apis with fastapi"
    cooking = "chef restaurant kitchen menu planning and food safety"
    score = scorer.score(software, cooking)
    assert score < 0.15


def test_score_is_deterministic(scorer: TFIDFScorer) -> None:
    a = "machine learning engineer tensorflow pytorch models"
    b = "data scientist python pandas statistics visualization"
    results = [scorer.score(a, b) for _ in range(5)]
    assert len(set(results)) == 1  # byte-identical every call


def test_score_always_bounded(scorer: TFIDFScorer) -> None:
    for a, b in [
        ("", ""),
        ("python", ""),
        ("python", "python"),
        ("java", "qwertyuiop nonsense token"),
    ]:
        score = scorer.score(a, b)
        assert 0.0 <= score <= 1.0


def test_persistence_roundtrip_identical_scores(tmp_path: Path) -> None:
    model_path = tmp_path / "vec.joblib"
    original = TFIDFScorer(model_path=model_path)
    original.fit(SYNTHETIC_CORPUS)
    a = "python developer django flask"
    b = "python engineer fastapi postgresql"
    original_score = original.score(a, b)

    # Fresh instance loads the persisted vectorizer from disk.
    reloaded = TFIDFScorer(model_path=model_path)
    assert reloaded.is_fitted
    assert reloaded.score(a, b) == original_score


def test_unfitted_scorer_raises(tmp_path: Path) -> None:
    s = TFIDFScorer(model_path=tmp_path / "missing.joblib")
    assert not s.is_fitted
    with pytest.raises(RuntimeError):
        s.score("a", "b")


def test_prepare_resume_text_assembles_expected_fields() -> None:
    resume = ParsedResume(
        raw_text="x",
        skills=["Python", "SQL"],
        experience=[ExperienceEntry(description="Built data pipelines in Python.")],
        education=[
            EducationEntry(degree="Bachelor's", field_of_study="Computer Science")
        ],
        total_years_experience=3.0,
        contact_info_present=True,
        parsing_confidence=0.9,
        parsing_warnings=[],
        pipeline_version="parser-v1",
    )
    text = prepare_resume_text_for_scoring(resume)
    assert "Python" in text and "SQL" in text
    assert "Built data pipelines" in text
    assert "Computer Science" in text and "Bachelor's" in text


def test_prepare_jd_text_uses_required_and_preferred_skills() -> None:
    jd = ParsedJobDescription(
        raw_text="x",
        required_skills=["Python", "SQL"],
        preferred_skills=["Docker"],
        required_years_experience=5.0,
        required_education_level="Bachelor's",
        parsing_confidence=0.8,
        pipeline_version="parser-v1",
    )
    text = prepare_jd_text_for_scoring(jd)
    assert "Python" in text and "SQL" in text and "Docker" in text

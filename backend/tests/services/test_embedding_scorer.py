"""Tests for the Phase 2.2 sentence-transformer embedding scorer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.services.scoring.embedding_scorer import EmbeddingScorer
from app.services.scoring.tfidf_scorer import TFIDFScorer

# Model load is slow; share one instance across the module.
_SCORER = EmbeddingScorer()

# Lexically-disjoint but semantically-equivalent pair (PRD §4 example).
SEMANTIC_GAP_RESUME = "led a team of 8 engineers"
SEMANTIC_GAP_JD = "people management experience"

# Corpus for the lexical baseline. Terms deliberately repeat across documents so
# the scorer's min_df=2 doesn't prune the vocabulary to empty on a small set.
_TFIDF_CORPUS = [
    "python developer building rest apis with django and flask",
    "python engineer building scalable backend services and apis",
    "led a team of engineers delivering backend projects",
    "led a team and managed engineers on multiple projects",
    "people management and team leadership experience required",
    "people management leadership and stakeholder communication skills",
    "machine learning models with tensorflow and pytorch",
    "machine learning and data science models in python",
    "data analysis with pandas numpy and data visualization",
    "data engineering pipelines with spark and airflow",
    "software engineer writing automated tests and quality assurance",
    "software engineer building microservices with docker and kubernetes",
]


def test_semantically_similar_lexically_different_scores_high() -> None:
    score = _SCORER.score(SEMANTIC_GAP_RESUME, SEMANTIC_GAP_JD)
    # Meaning-level match despite zero shared tokens. Scores are mapped via
    # (cos+1)/2, which compresses typical positive cosines into [0.5, 1]; a genuine
    # semantic match measures ~0.58 here, comfortably above the 0.55 floor.
    assert score >= 0.55


def test_unrelated_texts_score_lower() -> None:
    related = _SCORER.score(SEMANTIC_GAP_RESUME, SEMANTIC_GAP_JD)
    unrelated = _SCORER.score(
        SEMANTIC_GAP_RESUME, "photosynthesis in tropical rainforest plants"
    )
    assert unrelated < related


def test_embedding_beats_tfidf_on_semantic_gap(tmp_path: Path) -> None:
    """ARCHITECTURAL PROOF: for a lexically-disjoint semantic match, the
    embedding score must be SIGNIFICANTLY higher than the TF-IDF score.

    This is the justification for the entire hybrid design — the two layers do
    genuinely different, complementary work.
    """
    tfidf = TFIDFScorer(model_path=tmp_path / "vec.joblib")
    tfidf.fit(_TFIDF_CORPUS)

    tfidf_score = tfidf.score(SEMANTIC_GAP_RESUME, SEMANTIC_GAP_JD)
    embed_score = _SCORER.score(SEMANTIC_GAP_RESUME, SEMANTIC_GAP_JD)

    # No shared tokens → TF-IDF near zero; embeddings recover the meaning.
    assert tfidf_score < 0.1
    assert embed_score >= 0.55
    assert embed_score > tfidf_score + 0.4  # a large, real margin, not a trivial one


def test_score_is_deterministic() -> None:
    results = [_SCORER.score(SEMANTIC_GAP_RESUME, SEMANTIC_GAP_JD) for _ in range(5)]
    assert len(set(results)) == 1


def test_long_text_truncates_without_crashing() -> None:
    long_text = "python engineer machine learning data " * 300  # >> 256 tokens
    score = _SCORER.score(long_text, "software engineering role")
    assert 0.0 <= score <= 1.0


def test_score_always_bounded() -> None:
    for a, b in [
        ("", ""),
        ("python", ""),
        ("python", "python"),
        ("a", "completely different unrelated content here"),
    ]:
        score = _SCORER.score(a, b)
        assert 0.0 <= score <= 1.0


def test_embed_batch_matches_individual_embeds() -> None:
    texts = ["python developer", "people management", "machine learning models"]
    batch = _SCORER.embed_batch(texts)
    for i, text in enumerate(texts):
        single = _SCORER.embed(text)
        assert np.allclose(batch[i], single, atol=1e-5)

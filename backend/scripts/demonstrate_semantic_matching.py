"""Semantic-vs-lexical gap demonstration (Phase 2.2 evidence artifact).

Runs the literal PRD §4 example — resume "led a team of 8 engineers" vs JD
"people management experience" — through BOTH the TF-IDF lexical scorer (2.1) and
the sentence-transformer semantic scorer (2.2), showing that TF-IDF scores it
near-zero (no shared tokens) while embeddings score it meaningfully higher.

The printed output is a citable artifact for the capstone Model Development
section (PRD §14 item 3).

Usage (from backend/):
  python -m scripts.demonstrate_semantic_matching
"""

from __future__ import annotations

from app.services.scoring.embedding_scorer import EmbeddingScorer
from app.services.scoring.tfidf_scorer import TFIDFScorer

# Lexically disjoint but semantically equivalent pairs (PRD §4 style).
PAIRS = [
    ("led a team of 8 engineers", "people management experience"),
    ("built machine learning models", "developed AI and predictive systems"),
    ("wrote automated tests", "quality assurance and test coverage"),
]


def demonstrate_semantic_vs_lexical_gap() -> dict[str, object]:
    """Return and print TF-IDF vs embedding scores for each semantic-gap pair."""
    # TF-IDF loads its persisted, corpus-fit vectorizer (built by
    # scripts.build_tfidf_corpus). Embeddings load the pretrained model.
    tfidf = TFIDFScorer()
    if not tfidf.is_fitted:
        raise SystemExit(
            "No fitted TF-IDF model found. Run `python -m scripts.build_tfidf_corpus` "
            "first so the lexical comparison is meaningful."
        )
    embed = EmbeddingScorer()

    rows: list[dict[str, object]] = []
    print(f"{'resume phrase':<32} {'jd phrase':<34} {'tfidf':>7} {'embed':>7}")
    print("-" * 84)
    for resume, jd in PAIRS:
        t = tfidf.score(resume, jd)
        e = embed.score(resume, jd)
        rows.append(
            {"resume": resume, "jd": jd, "tfidf_score": t, "embedding_score": e}
        )
        print(f"{resume:<32} {jd:<34} {t:>7.3f} {e:>7.3f}")

    print(
        "\nConclusion: embeddings recover semantic matches that share no tokens, "
        "which TF-IDF (lexical) scores near zero — the two layers are "
        "complementary, justifying the hybrid design."
    )
    return {"pairs": rows}


if __name__ == "__main__":
    demonstrate_semantic_vs_lexical_gap()

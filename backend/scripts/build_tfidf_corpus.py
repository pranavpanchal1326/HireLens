"""Build and persist the TF-IDF corpus + fitted vectorizer (Phase 2.1).

Run ONCE after downloading the Kaggle datasets. Corpus building is slow and
should not repeat on every app startup — this persists both the corpus and the
fitted vectorizer to data/processed/ for fast reuse.

Usage (from repo root):
  python -m scripts.build_tfidf_corpus
  python -m scripts.build_tfidf_corpus --max-resumes 2000 --max-jds 20000

The --max-* flags fit an honest proof-of-concept-scale sample (PRD §7.3) instead
of the full ~124k JD corpus, which is often the sensible default for a capstone.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.services.scoring.corpus_builder import CorpusLoader, save_corpus
from app.services.scoring.tfidf_scorer import TFIDFScorer

# backend/scripts/build_tfidf_corpus.py -> parents[2] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RAW = _REPO_ROOT / "data" / "raw"
_PROCESSED = _REPO_ROOT / "data" / "processed"

DEFAULT_RESUME_CSV = _RAW / "resume" / "Resume.csv"
DEFAULT_JD_CSV = _RAW / "jd" / "postings.csv"
CORPUS_OUT = _PROCESSED / "tfidf_corpus.pkl"
MODEL_OUT = _PROCESSED / "tfidf_vectorizer.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the TF-IDF corpus + model.")
    parser.add_argument("--resume-csv", default=str(DEFAULT_RESUME_CSV))
    parser.add_argument("--jd-csv", default=str(DEFAULT_JD_CSV))
    parser.add_argument("--max-resumes", type=int, default=None)
    parser.add_argument("--max-jds", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print(f"Loading corpus...\n  resumes: {args.resume_csv}\n  jds:     {args.jd_csv}")
    loader = CorpusLoader()
    corpus = loader.load_and_prepare_corpus(
        resume_csv_path=args.resume_csv,
        jd_csv_path=args.jd_csv,
        max_resumes=args.max_resumes,
        max_jds=args.max_jds,
    )
    print(f"Corpus: {len(corpus)} unique documents")

    save_corpus(corpus, str(CORPUS_OUT))
    print(f"Saved corpus -> {CORPUS_OUT}")

    print("Fitting TF-IDF vectorizer...")
    scorer = TFIDFScorer(model_path=MODEL_OUT)
    scorer.fit(corpus)
    assert scorer._vectorizer is not None
    print(
        f"Vocabulary size: {len(scorer._vectorizer.vocabulary_)}\n"
        f"Saved vectorizer -> {MODEL_OUT}"
    )
    print("Done.")


if __name__ == "__main__":
    main()

"""TF-IDF corpus construction (Phase 2.1).

Builds a unified, cleaned document corpus from the Kaggle Resume Dataset and the
Kaggle LinkedIn Job Postings dataset (PRD §6). TF-IDF's IDF term is only
statistically meaningful over a real corpus — this module produces that corpus so
the vectorizer is NEVER fit ad-hoc on the two documents being compared at request
time.

Persistence uses pickle because real resume/JD text contains embedded newlines,
which a naive one-document-per-line text file cannot round-trip safely.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Default text columns (verified against the on-disk Kaggle files). Configurable
# so a differently-shaped export doesn't require code changes.
DEFAULT_RESUME_COLUMN = "Resume_str"
DEFAULT_JD_COLUMN = "description"

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


class CorpusLoader:
    """Loads and prepares the unified resume + JD corpus for TF-IDF fitting."""

    def load_and_prepare_corpus(
        self,
        resume_csv_path: str,
        jd_csv_path: str,
        resume_column: str = DEFAULT_RESUME_COLUMN,
        jd_column: str = DEFAULT_JD_COLUMN,
        max_resumes: int | None = None,
        max_jds: int | None = None,
    ) -> list[str]:
        """Return a deduplicated list of cleaned documents from both datasets.

        Resumes and JDs are combined into ONE corpus so IDF sees the full
        vocabulary space of both document types being compared. ``max_*`` allow a
        smaller, honest proof-of-concept-scale sample (PRD §7.3) when desired.
        """
        resume_docs = self._load_column(resume_csv_path, resume_column, max_resumes)
        jd_docs = self._load_column(jd_csv_path, jd_column, max_jds)
        logger.info(
            "Loaded %d resume docs and %d JD docs", len(resume_docs), len(jd_docs)
        )

        combined = resume_docs + jd_docs
        cleaned = [self._clean(doc) for doc in combined]
        # Drop empties produced by cleaning, then dedupe preserving order.
        deduped = list(dict.fromkeys(d for d in cleaned if d))
        logger.info("Corpus prepared: %d unique documents", len(deduped))
        return deduped

    def _load_column(self, csv_path: str, column: str, limit: int | None) -> list[str]:
        """Read a single text column, skipping missing/malformed rows gracefully."""
        docs: list[str] = []
        # Chunked read keeps memory bounded on the ~500MB JD file.
        reader = pd.read_csv(
            csv_path,
            usecols=lambda c: c == column,
            chunksize=10_000,
            dtype=str,
            on_bad_lines="skip",
        )
        for chunk in reader:
            if column not in chunk.columns:
                raise KeyError(
                    f"Column {column!r} not found in {csv_path}. "
                    f"Available handling: pass the correct *_column argument."
                )
            for value in chunk[column].tolist():
                if not isinstance(value, str) or not value.strip():
                    continue  # Skip NaN / empty rows without crashing the build.
                docs.append(value)
                if limit is not None and len(docs) >= limit:
                    return docs
        return docs

    def _clean(self, text: str) -> str:
        """Light cleaning only. scikit-learn handles stopwords/tokenization — we do
        NOT strip punctuation or stopwords here (that would duplicate its work)."""
        text = _HTML_TAG_RE.sub(" ", text)  # LinkedIn scrape often has HTML.
        text = text.lower()
        text = _WS_RE.sub(" ", text).strip()
        return text


def save_corpus(corpus: list[str], output_path: str) -> None:
    """Persist a prepared corpus (pickle — safe with embedded newlines)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_corpus(input_path: str) -> list[str]:
    """Load a previously-persisted corpus."""
    with Path(input_path).open("rb") as f:
        corpus: list[str] = pickle.load(f)
    return corpus

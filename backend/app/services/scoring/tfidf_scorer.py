"""TF-IDF lexical scorer (Phase 2.1).

Produces the ``tfidf_score`` field of the locked FeatureVector (Phase 0.2) — a
single float in [0.0, 1.0] measuring lexical overlap between a resume and a JD.
This is the ONE "trained by you" statistical layer (PRD §5): the vectorizer is
fit on YOUR corpus, persisted, and reused — never fit ad-hoc per request.

Standalone by design: this output is meaningful with zero other scoring
components active, which is what makes pipeline version ``v1-tfidf`` an honest,
isolated ablation stage (Phase 0.3 / PRD §7.2).
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.schemas.parsing import ParsedJobDescription, ParsedResume
from app.services.scoring.corpus_builder import load_corpus

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
DEFAULT_MODEL_PATH = _PROCESSED_DIR / "tfidf_vectorizer.joblib"

_SCORE_PRECISION = 6


class TFIDFScorer:
    """Fits/loads a corpus-wide TF-IDF vectorizer and scores resume vs JD text."""

    def __init__(
        self, corpus_path: str | None = None, model_path: str | Path | None = None
    ) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._vectorizer: TfidfVectorizer | None = None

        if self.model_path.exists():
            # Reuse the persisted, already-fit vectorizer — no re-fit on startup.
            self._vectorizer = joblib.load(self.model_path)
            logger.info("Loaded fitted TF-IDF vectorizer from %s", self.model_path)
        elif corpus_path is not None:
            self.fit(load_corpus(corpus_path))

    @property
    def is_fitted(self) -> bool:
        return self._vectorizer is not None

    def fit(self, corpus: list[str]) -> None:
        """Fit the vectorizer on the full corpus and persist it via joblib.

        Parameter choices (each deliberate):
          - stop_words="english": native, appropriate stopword removal; we do not
            hand-roll a duplicate stoplist.
          - ngram_range=(1, 2): unigrams alone would split multi-word skills like
            "machine learning" or "data science" into unrelated tokens, losing the
            phrase signal that matters most for resume/JD matching; bigrams capture
            them.
          - max_features=10000: bounds the vocabulary so the ~120k-doc JD corpus
            doesn't explode memory/compute; 10k covers the high-signal terms while
            discarding the long tail of rare noise.
          - min_df=2: ignore terms appearing in only one document — filters typos
            and one-off noise from the large scraped LinkedIn dataset.
        TF-IDF is deterministic: vocabulary is built from the corpus content and
        ordered stably, so identical input always yields identical vectors.
        """
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000,
            min_df=2,
        )
        vectorizer.fit(corpus)
        self._vectorizer = vectorizer

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, self.model_path)
        logger.info(
            "Fitted TF-IDF vectorizer (vocab=%d) and saved to %s",
            len(vectorizer.vocabulary_),
            self.model_path,
        )

    def _vectorize(self, text: str) -> csr_matrix:
        if self._vectorizer is None:
            raise RuntimeError(
                "TFIDFScorer is not fitted. Provide a corpus_path/model_path or "
                "call fit() before scoring."
            )
        return self._vectorizer.transform([text])

    def score(self, resume_text: str, jd_text: str) -> float:
        """Cosine similarity of the two TF-IDF vectors, as a float in [0.0, 1.0].

        Cosine similarity of non-negative TF-IDF vectors is already bounded to
        [0, 1]; the clip is a determinism/robustness safety net (guards against
        tiny floating-point overshoot), NOT a correction of expected behavior.
        Empty/no-overlap inputs yield 0.0.
        """
        resume_vec = self._vectorize(resume_text)
        jd_vec = self._vectorize(jd_text)
        similarity = float(cosine_similarity(resume_vec, jd_vec)[0][0])
        clipped = min(1.0, max(0.0, similarity))
        return round(clipped, _SCORE_PRECISION)


def prepare_resume_text_for_scoring(parsed_resume: ParsedResume) -> str:
    """Assemble the resume fields most relevant to lexical fit into one blob.

    Included (and why): skills (the primary lexical signal), experience entry
    descriptions (role responsibilities/technologies in prose), and education
    degree + field_of_study (domain signal). Raw contact info is intentionally
    excluded (PII, and no scoring value).
    """
    parts: list[str] = []
    parts.extend(parsed_resume.skills)
    parts.extend(entry.description for entry in parsed_resume.experience)
    for edu in parsed_resume.education:
        if edu.degree:
            parts.append(edu.degree)
        if edu.field_of_study:
            parts.append(edu.field_of_study)
    return " ".join(p for p in parts if p).strip()


def prepare_jd_text_for_scoring(parsed_jd: ParsedJobDescription) -> str:
    """Assemble JD fields for lexical fit: required + preferred skills.

    These are the fields a resume is actually matched against; the surrounding
    boilerplate of a posting adds noise rather than fit signal.
    """
    parts: list[str] = [*parsed_jd.required_skills, *parsed_jd.preferred_skills]
    return " ".join(p for p in parts if p).strip()

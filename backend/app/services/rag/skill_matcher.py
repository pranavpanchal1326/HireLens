"""Semantic skill-matching decision logic (Phase 3.3).

Turns Phase 3.2's raw vector retrieval into real match/gap determinations:
which resume skills match which JD requirements, HOW (exact string / taxonomy
synonym / semantic RAG), the skill_overlap_pct feature, and action-phrased gaps.
Directly drives Design Blueprint §10.6's exact-vs-≈ UI and §12's blameless voice.

Scope: matching decisions only. NO similar-case lookup (3.4), NO orchestration (4).
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.scoring import GapItem, SkillMatch
from app.services.rag.faiss_index_builder import FAISSSkillIndexQuerier
from app.services.rag.taxonomy_schemas import SkillTaxonomyEntry
from app.services.rag.text_normalization import normalize_skill_text

# --- Provisional, tunable constants (Phase 5/6 will validate these) ----------
# Minimum FAISS cosine similarity to count a semantic match as REAL, not noise.
# Empirically (Phase 3.2), genuine transferable matches land ~0.5 (e.g. "led a
# team of engineers" ≈ "team leadership" = 0.52) while unrelated pairs sit ~0.2.
# 0.45 separates signal from noise: lower risks dishonest false-positive matches
# (Design Blueprint P3); higher would miss genuine transferable skills (defeats
# PRD §4). PROVISIONAL — pending ground-truth validation in Phase 5, mirroring
# DEFAULT_HYBRID_WEIGHTS's honesty pattern.
SEMANTIC_MATCH_THRESHOLD = 0.45

# skill_overlap_pct weighting: a matched REQUIRED skill counts fully; a matched
# PREFERRED (nice-to-have) skill counts half — missing a hard requirement should
# hurt overlap more than missing a bonus. PROVISIONAL — pending Phase 6 tuning.
_REQUIRED_WEIGHT = 1.0
_PREFERRED_WEIGHT = 0.5

_TOP_K = 5
_OVERLAP_PRECISION = 6


class SkillMatchResult(BaseModel):
    """Richer intermediate match record (converted to SkillMatch for output)."""

    resume_skill: str
    jd_skill: str
    match_type: Literal["exact", "semantic", "none"]
    similarity_score: float = Field(ge=0.0, le=1.0)
    matched_via_concept_uri: str | None = None


# Action-phrased gap templates — NEVER deficiency-framed (Design Blueprint §12).
_GAP_TEMPLATES = (
    "Highlight any experience you have with {skill}.",
    "Consider gaining exposure to {skill} to strengthen your profile.",
    "Add specific examples of {skill} if you have worked with it.",
    "If you've used {skill}, make it visible on your resume.",
)


def generate_action_phrased_gap(missing_skill: str) -> str:
    """Produce an action-framed suggestion for a missing skill.

    Deterministic template selection (so output is stable per skill) while still
    varying across skills. Uses a STABLE hash (not Python's per-process-randomized
    ``hash()``) so the choice is reproducible across runs. Content rule: never
    deficiency-framed language.
    """
    digest = hashlib.sha1(normalize_skill_text(missing_skill).encode("utf-8"))
    idx = int(digest.hexdigest(), 16) % len(_GAP_TEMPLATES)
    return _GAP_TEMPLATES[idx].format(skill=missing_skill)


class SkillMatcher:
    """Decides exact / synonym / semantic / no-match between resume and JD skills."""

    def __init__(self, faiss_querier: FAISSSkillIndexQuerier) -> None:
        self._querier = faiss_querier
        self._label_to_uris: dict[str, set[str]] = {}
        self._indexed_id: int | None = None

    def _ensure_label_index(self, taxonomy_entries: list[SkillTaxonomyEntry]) -> None:
        """Build (once, cached by identity) a normalized-label → {concept_uri} map."""
        if self._indexed_id == id(taxonomy_entries):
            return
        label_to_uris: dict[str, set[str]] = {}
        for entry in taxonomy_entries:
            for label in [entry.preferred_label, *entry.alt_labels]:
                key = normalize_skill_text(label)
                if key:
                    label_to_uris.setdefault(key, set()).add(entry.concept_uri)
        self._label_to_uris = label_to_uris
        self._indexed_id = id(taxonomy_entries)

    def _find_exact_match(self, resume_skill: str, jd_skill: str) -> bool:
        """Normalized direct string equality."""
        return normalize_skill_text(resume_skill) == normalize_skill_text(jd_skill)

    def _find_synonym_match(
        self,
        resume_skill: str,
        jd_skill: str,
        taxonomy_entries: list[SkillTaxonomyEntry],
    ) -> str | None:
        """Return a shared concept_uri if both skills are known labels of the SAME
        ESCO concept (a precise taxonomy lookup, not a fuzzy guess); else None."""
        self._ensure_label_index(taxonomy_entries)
        resume_uris = self._label_to_uris.get(normalize_skill_text(resume_skill), set())
        jd_uris = self._label_to_uris.get(normalize_skill_text(jd_skill), set())
        shared = resume_uris & jd_uris
        return next(iter(shared)) if shared else None

    def _concepts_for_jd_skill(self, jd_skill: str) -> set[str]:
        """Concept URIs jd_skill belongs to: known labels first, else nearest via
        FAISS (so a JD skill not literally in the taxonomy can still be bridged)."""
        known = self._label_to_uris.get(normalize_skill_text(jd_skill), set())
        if known:
            return known
        nearest = self._querier.query_raw(jd_skill, top_k=1)
        if nearest and nearest[0][1] >= SEMANTIC_MATCH_THRESHOLD:
            return {nearest[0][0].concept_uri}
        return set()

    def _find_semantic_match(
        self, resume_skill: str, jd_skill: str
    ) -> tuple[bool, float, str | None]:
        """True if the resume skill retrieves jd_skill's concept above threshold."""
        jd_uris = self._concepts_for_jd_skill(jd_skill)
        if not jd_uris:
            return False, 0.0, None
        best_uri: str | None = None
        best_score = 0.0
        for entry, score in self._querier.query_raw(resume_skill, top_k=_TOP_K):
            if (
                score >= SEMANTIC_MATCH_THRESHOLD
                and entry.concept_uri in jd_uris
                and score > best_score
            ):
                best_uri, best_score = entry.concept_uri, score
        if best_uri is not None:
            return True, best_score, best_uri
        return False, 0.0, None

    def match_single_skill(
        self,
        resume_skill: str,
        jd_skill: str,
        taxonomy_entries: list[SkillTaxonomyEntry],
    ) -> SkillMatchResult:
        """Check exact → synonym → semantic, in that order.

        Ordering is both a performance optimization (cheap certain checks first,
        only paying for a FAISS query when they fail) AND a correctness rule: an
        exact/synonym match must never be downgraded to a lower-confidence semantic
        match just because semantic checking would also have run.
        """
        self._ensure_label_index(taxonomy_entries)

        if self._find_exact_match(resume_skill, jd_skill):
            return SkillMatchResult(
                resume_skill=resume_skill,
                jd_skill=jd_skill,
                match_type="exact",
                similarity_score=1.0,
            )

        shared_uri = self._find_synonym_match(resume_skill, jd_skill, taxonomy_entries)
        if shared_uri is not None:
            return SkillMatchResult(
                resume_skill=resume_skill,
                jd_skill=jd_skill,
                match_type="exact",  # a precise taxonomy synonym is exact, not fuzzy
                similarity_score=1.0,
                matched_via_concept_uri=shared_uri,
            )

        matched, score, uri = self._find_semantic_match(resume_skill, jd_skill)
        if matched:
            return SkillMatchResult(
                resume_skill=resume_skill,
                jd_skill=jd_skill,
                match_type="semantic",
                similarity_score=score,
                matched_via_concept_uri=uri,
            )
        return SkillMatchResult(
            resume_skill=resume_skill,
            jd_skill=jd_skill,
            match_type="none",
            similarity_score=0.0,
        )

    def _best_match_for_jd_skill(
        self,
        jd_skill: str,
        resume_skills: list[str],
        used_resume_skills: set[str],
        taxonomy_entries: list[SkillTaxonomyEntry],
    ) -> SkillMatchResult | None:
        """Best available match for one JD skill among UNUSED resume skills.

        Anti-double-counting: a resume skill already claimed by another JD skill is
        skipped, so one resume line can't inflate overlap across many requirements.
        Preference: exact over semantic, then higher similarity.
        """
        best: SkillMatchResult | None = None
        for resume_skill in resume_skills:
            if resume_skill in used_resume_skills:
                continue
            result = self.match_single_skill(resume_skill, jd_skill, taxonomy_entries)
            if result.match_type == "none":
                continue
            if best is None or _rank(result) > _rank(best):
                best = result
        return best

    def match_resume_to_jd(
        self,
        resume_skills: list[str],
        required_skills: list[str],
        preferred_skills: list[str],
        taxonomy_entries: list[SkillTaxonomyEntry],
    ) -> tuple[float, list[SkillMatch], list[GapItem]]:
        """Match all JD skills against resume skills; return
        (skill_overlap_pct, matched_skills, gaps)."""
        self._ensure_label_index(taxonomy_entries)
        used: set[str] = set()
        matched: list[SkillMatch] = []
        gaps: list[GapItem] = []
        matched_weight = 0.0

        for pool, weight in (
            (required_skills, _REQUIRED_WEIGHT),
            (preferred_skills, _PREFERRED_WEIGHT),
        ):
            for jd_skill in pool:
                best = self._best_match_for_jd_skill(
                    jd_skill, resume_skills, used, taxonomy_entries
                )
                if best is not None:
                    used.add(best.resume_skill)
                    matched_weight += weight
                    matched.append(
                        SkillMatch(
                            resume_skill=best.resume_skill,
                            jd_skill=best.jd_skill,
                            match_type=best.match_type,  # type: ignore[arg-type]
                            similarity_score=best.similarity_score,
                        )
                    )
                else:
                    gaps.append(
                        GapItem(
                            missing_skill=jd_skill,
                            suggested_action=generate_action_phrased_gap(jd_skill),
                        )
                    )

        total_weight = (
            len(required_skills) * _REQUIRED_WEIGHT
            + len(preferred_skills) * _PREFERRED_WEIGHT
        )
        overlap = matched_weight / total_weight if total_weight > 0 else 0.0
        return round(overlap, _OVERLAP_PRECISION), matched, gaps


def _rank(result: SkillMatchResult) -> tuple[int, float]:
    """Ordering key: exact (2) beats semantic (1); tie-break on similarity."""
    priority = 2 if result.match_type == "exact" else 1
    return priority, result.similarity_score

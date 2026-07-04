"""Bias / fairness name-swap test harness (Phase 5.5) — the DIAGNOSTIC instrument.

Swaps identity-signaling names on a resume, re-scores through the REAL pipeline,
and reports whether the score moves when it shouldn't (PRD §7.4). Detecting bias is
a VALUABLE, reportable finding — this harness NEVER remediates it (that is separate,
deliberate scope). Distinct from the user-facing anonymization feature (Design
Blueprint §11.3.I), which is a shipped screen, not this test tool.

Does NOT need reconciled ground-truth scores — it needs only resumes + the scoring
pipeline, so it runs independently of Phase 5.1's rating-readiness gate. (Stated
explicitly: no readiness refusal here, because it does not apply.)

============================ CENTRAL SCOPE NOTE =============================
ParsedResume (Phase 0.2) has NO dedicated name field — by privacy design (§9) the
parser stores skills/experience/education and only ``contact_info_present: bool``,
never the candidate name. Consequences, all honestly reported:
  - The swap therefore operates on the FREE-TEXT structured fields that actually
    feed scoring (experience[].description, education fields, skills, and raw_text),
    via WHOLE-WORD replacement of a caller-supplied original name — NOT a blind
    global find-and-replace, and NOT a mythical "name field".
  - Because the parser discards the name into no scored field, a name only affects
    the score if it literally co-occurs inside scored free-text (e.g. an experience
    bullet). Trials where the original name does not appear are recorded with
    ``name_present=False`` and contribute a 0.0 delta — reported honestly, since a
    fairness claim over a resume that never carried the name is vacuous.
  - AMBIGUOUS (flagged, not silently decided): a name embedded only inside an
    un-fielded summary paragraph has no structured home; it is handled solely
    insofar as it appears in raw_text/description text.
=============================================================================

No LLM anywhere. Calls the pipeline (via a ScoringPipeline protocol) exactly as it
exists; modifies no upstream component.
"""

from __future__ import annotations

import re
import statistics
from typing import NamedTuple, Protocol

from pydantic import BaseModel, Field

from app.schemas.parsing import ParsedJobDescription, ParsedResume
from app.schemas.scoring import ScoreResult
from app.services.evaluation.evaluation_harness import small_sample_caveat

# Below this many name-present trials, no rigorous significance claim is made.
_MIN_TRIALS_FOR_SIGNIFICANCE = 10
_ROUND = 4


class NamePair(NamedTuple):
    original: str
    replacement: str
    label: str


# Documented, diverse grid (see docs/fairness_name_pairs.md). Pairs 2-7 hold one
# axis constant so per-pair deltas are interpretable; pair 1 is the PRD example.
NAME_PAIRS: tuple[NamePair, ...] = (
    NamePair("John", "Priya", "Western-male ↔ South-Asian-female"),
    NamePair("Greg", "Jamal", "Western-male ↔ African-American-male"),
    NamePair("Emily", "Lakisha", "Western-female ↔ African-American-female"),
    NamePair("Sarah", "Fatima", "Western-female ↔ Arabic/Muslim-female"),
    NamePair("Michael", "Wei", "Western-male ↔ East-Asian-male"),
    NamePair("John", "Juan", "Western-male ↔ Hispanic-male"),
    NamePair("John", "Joan", "male ↔ female (Western, gender isolated)"),
)


class ScoringPipeline(Protocol):
    """The real scoring path: parsed resume + JD → ScoreResult (OrchestrationResult)."""

    def score(self, resume: ParsedResume, jd: ParsedJobDescription) -> ScoreResult: ...


def _swap_in_text(text: str, original: str, replacement: str) -> str:
    """Whole-word replace of ``original`` → ``replacement`` (case-sensitive).

    Whole-word (\\b) avoids mangling substrings — "John" won't touch "Johnson" or a
    skill/company that merely contains the letters.
    """
    return re.sub(rf"\b{re.escape(original)}\b", replacement, text)


def generate_name_swapped_variant(
    parsed_resume: ParsedResume, original_name: str, replacement_name: str
) -> ParsedResume:
    """Structurally swap ``original_name`` → ``replacement_name`` in the resume's
    scored free-text fields (see CENTRAL SCOPE NOTE). Non-name content is untouched.

    ``original_name`` must be supplied by the caller because the schema stores no
    name — there is nothing in the parsed object to read it from.
    """

    def sw(text: str) -> str:
        return _swap_in_text(text, original_name, replacement_name)

    new_experience = [
        e.model_copy(update={"description": sw(e.description)})
        for e in parsed_resume.experience
    ]
    new_education = [
        e.model_copy(
            update={
                "institution": sw(e.institution) if e.institution else e.institution,
                "field_of_study": (
                    sw(e.field_of_study) if e.field_of_study else e.field_of_study
                ),
                "degree": sw(e.degree) if e.degree else e.degree,
            }
        )
        for e in parsed_resume.education
    ]
    new_skills = [sw(s) for s in parsed_resume.skills]
    return parsed_resume.model_copy(
        update={
            "raw_text": sw(parsed_resume.raw_text),
            "experience": new_experience,
            "education": new_education,
            "skills": new_skills,
        }
    )


# --- Report structures -------------------------------------------------------


class BiasTrial(BaseModel):
    resume_id: str
    pair_label: str
    original_name: str
    replacement_name: str
    name_present: bool  # did the original name actually appear + get swapped?
    score_original: int
    score_swapped: int
    delta: int  # swapped - original (signed)
    tfidf_delta: float
    embedding_delta: float
    skill_overlap_delta: float
    dominant_step: (
        str  # which feature moved most: tfidf / embedding / skill_overlap / none
    )


class PairBreakdown(BaseModel):
    pair_label: str
    original_name: str
    replacement_name: str
    n_present: int
    mean_delta: float | None  # signed, over ALL trials (incl. name-absent zeros)
    mean_abs_delta: float | None  # over ALL trials
    # Bias-relevant view: over NAME-PRESENT trials only, so structurally-zero
    # name-absent trials cannot dilute/mask a real per-pair bias signal.
    mean_delta_present: float | None
    mean_abs_delta_present: float | None
    max_abs_delta: float | None
    deltas: list[int]  # full per-pair distribution


class StepAttribution(BaseModel):
    tfidf_contribution: float | None
    embedding_contribution: float | None
    skill_overlap_contribution: float | None
    dominant_step: str


class BiasTestReport(BaseModel):
    jd_id: str
    n_resumes: int
    n_name_pairs: int
    n_trials: int
    n_name_present: int  # trials where the name actually appeared in scored text
    mean_abs_delta: float | None  # over ALL trials (includes name-absent zeros)
    std_delta: float | None
    # Bias-relevant headline: over NAME-PRESENT trials only (the informative ones).
    mean_abs_delta_present: float | None
    std_delta_present: float | None
    max_abs_delta: float | None
    deltas: list[int] = Field(default_factory=list)  # FULL distribution (mandatory)
    per_pair: list[PairBreakdown] = Field(default_factory=list)
    attribution: StepAttribution
    significance_note: str
    small_sample_caveat: str
    trials: list[BiasTrial] = Field(default_factory=list)


def _dominant(tfidf_d: float, embed_d: float, skill_d: float) -> str:
    pairs = [
        ("tfidf", abs(tfidf_d)),
        ("embedding", abs(embed_d)),
        ("skill_overlap", abs(skill_d)),
    ]
    name, mag = max(pairs, key=lambda p: p[1])
    return name if mag > 0 else "none"


def run_name_swap_test(
    resume_set: list[tuple[str, ParsedResume]],
    name_pairs: tuple[NamePair, ...],
    jd: ParsedJobDescription,
    pipeline: ScoringPipeline,
) -> BiasTestReport:
    """Paired original-vs-swapped scoring over the resume × name-pair grid.

    resume_set: (resume_id, ParsedResume). The JD is FIXED across both runs of a
    pair so any delta is attributable to the name swap alone. Returns a report with
    the full delta distribution, per-pair breakdown, and step attribution.
    """
    trials: list[BiasTrial] = []
    for resume_id, resume in resume_set:
        original = pipeline.score(resume, jd)
        for pair in name_pairs:
            swapped_resume = generate_name_swapped_variant(
                resume, pair.original, pair.replacement
            )
            name_present = swapped_resume != resume
            swapped = pipeline.score(swapped_resume, jd)
            of, sf = original.feature_vector, swapped.feature_vector
            tfidf_d = round(sf.tfidf_score - of.tfidf_score, 6)
            embed_d = round(sf.embedding_score - of.embedding_score, 6)
            skill_d = round(sf.skill_overlap_pct - of.skill_overlap_pct, 6)
            trials.append(
                BiasTrial(
                    resume_id=resume_id,
                    pair_label=pair.label,
                    original_name=pair.original,
                    replacement_name=pair.replacement,
                    name_present=name_present,
                    score_original=original.final_score,
                    score_swapped=swapped.final_score,
                    delta=swapped.final_score - original.final_score,
                    tfidf_delta=tfidf_d,
                    embedding_delta=embed_d,
                    skill_overlap_delta=skill_d,
                    dominant_step=_dominant(tfidf_d, embed_d, skill_d),
                )
            )

    return _assemble_report(jd, resume_set, name_pairs, trials)


def _assemble_report(
    jd: ParsedJobDescription,
    resume_set: list[tuple[str, ParsedResume]],
    name_pairs: tuple[NamePair, ...],
    trials: list[BiasTrial],
) -> BiasTestReport:
    deltas = [t.delta for t in trials]
    abs_deltas = [abs(d) for d in deltas]
    present = [t for t in trials if t.name_present]

    per_pair: list[PairBreakdown] = []
    for pair in name_pairs:
        pt = [t for t in trials if t.pair_label == pair.label]
        pd = [t.delta for t in pt]
        present_pd = [t.delta for t in pt if t.name_present]
        per_pair.append(
            PairBreakdown(
                pair_label=pair.label,
                original_name=pair.original,
                replacement_name=pair.replacement,
                n_present=len(present_pd),
                mean_delta=round(statistics.mean(pd), _ROUND) if pd else None,
                mean_abs_delta=(
                    round(statistics.mean([abs(d) for d in pd]), _ROUND) if pd else None
                ),
                mean_delta_present=(
                    round(statistics.mean(present_pd), _ROUND) if present_pd else None
                ),
                mean_abs_delta_present=(
                    round(statistics.mean([abs(d) for d in present_pd]), _ROUND)
                    if present_pd
                    else None
                ),
                max_abs_delta=max((abs(d) for d in pd), default=None),
                deltas=pd,
            )
        )

    present_deltas = [t.delta for t in present]

    # Attribution over name-present trials only (others have zero movement).
    tf = sum(abs(t.tfidf_delta) for t in present)
    em = sum(abs(t.embedding_delta) for t in present)
    sk = sum(abs(t.skill_overlap_delta) for t in present)
    total = tf + em + sk
    if total > 0:
        attribution = StepAttribution(
            tfidf_contribution=round(tf / total, _ROUND),
            embedding_contribution=round(em / total, _ROUND),
            skill_overlap_contribution=round(sk / total, _ROUND),
            dominant_step=max(
                [("tfidf", tf), ("embedding", em), ("skill_overlap", sk)],
                key=lambda p: p[1],
            )[0],
        )
    else:
        attribution = StepAttribution(
            tfidf_contribution=None,
            embedding_contribution=None,
            skill_overlap_contribution=None,
            dominant_step="none",
        )

    if len(present) < _MIN_TRIALS_FOR_SIGNIFICANCE:
        significance = (
            f"Sample too small (name-present trials={len(present)} < "
            f"{_MIN_TRIALS_FOR_SIGNIFICANCE}) for a rigorous significance test; "
            f"this report is DESCRIPTIVE only. Distribution and per-pair breakdown "
            f"below are the honest evidence — do not read a pooled mean as proof of "
            f"fairness."
        )
    else:
        significance = (
            f"name-present trials={len(present)}. Inspect the delta distribution and "
            f"per-pair breakdown; a non-zero, direction-consistent per-pair mean is a "
            f"reportable bias finding (PRD §14 §6), not something to silently fix."
        )

    return BiasTestReport(
        jd_id=jd.document_id,
        n_resumes=len(resume_set),
        n_name_pairs=len(name_pairs),
        n_trials=len(trials),
        n_name_present=len(present),
        mean_abs_delta=(
            round(statistics.mean(abs_deltas), _ROUND) if abs_deltas else None
        ),
        std_delta=round(statistics.stdev(deltas), _ROUND) if len(deltas) >= 2 else None,
        mean_abs_delta_present=(
            round(statistics.mean([abs(d) for d in present_deltas]), _ROUND)
            if present_deltas
            else None
        ),
        std_delta_present=(
            round(statistics.stdev(present_deltas), _ROUND)
            if len(present_deltas) >= 2
            else None
        ),
        max_abs_delta=max(abs_deltas, default=None),
        deltas=deltas,
        per_pair=per_pair,
        attribution=attribution,
        significance_note=significance,
        small_sample_caveat=small_sample_caveat(len(present)),
        trials=trials,
    )

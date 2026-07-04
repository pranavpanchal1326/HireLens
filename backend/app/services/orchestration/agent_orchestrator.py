"""Agent Orchestrator — deterministic control flow, STEP 1-2 wired (Phase 4.2).

=====================================================================================
ARCHITECTURAL CONSTRAINT (NON-NEGOTIABLE, PRD §4 / §15):
DETERMINISTIC, RULE-BASED control flow. NOT an LLM agent. ZERO LLM calls, prompts,
chain-of-thought, or natural-language decisions anywhere. Every routing choice is
plain, auditable if/else code. (The only permitted LLM is the optional downstream
polish layer — out of scope here, absent from this file.)
=====================================================================================

=============================== TRACEABILITY ================================
UPSTREAM (fixed — imported/called, never redefined):
  - Data contracts (ParsedResume, ParsedJobDescription, FeatureVector,
    ScoreResult, SkillMatch, GapItem) ............ Phase 0.2
  - Pipeline versioning (locked enum + registry) . Phase 0.3
  - Confidence banding utility ................... Phase 1.3
  - Hybrid Scorer (STEP 1) ....................... Phase 2.4  [WIRED]
  - RAG Skill Matcher (STEP 2) ................... Phase 3.3  [WIRED]
STEP STATUS (ALL LIVE — run_orchestration is end-to-end as of Phase 4.4):
  - STEP 1 hybrid_scoring ........................ LIVE (Phase 4.2)
  - STEP 2 rag_skill_matching .................... LIVE (Phase 4.2)
  - STEP 3 experience_years_matching ............. LIVE (Phase 4.4; matcher module
                                                   authored in 4.4 to unblock — flagged)
  - STEP 4 ambiguity_flagging .................... LIVE (Phase 4.3)
  - STEP 5 decision_logic (compute_final_decision) LIVE (Phase 4.4)

CHAIN OF CUSTODY for one run (verify no explainability data is silently dropped):
  parsed_resume + parsed_jd
    → STEP 1 hybrid ScoreResult (tfidf, embedding)     [state.hybrid_result]
    → STEP 2 skill_overlap + matched_skills + gaps     [state.skill_overlap_pct/…]
    → STEP 3 exp_match                                 [state.exp_match]
    → STEP 4 AmbiguityFlag (reasons, advisory band)    [state.ambiguity_flag]
    → STEP 5 compute_final_decision → ScoreResult (final_score, final band,
             merged matched_skills + gaps, carried pipeline_version)

SCHEMA GAP — RESOLVED (post-4.4 hardening): ScoreResult (Phase 0.2) gained an
additive, back-compatible `confidence_reasons: list[str]` field. STEP 5 now merges
STEP 4's AmbiguityFlag.reasons with its own decision-layer reasons (UNION, each
traceable) into that field — the §10.8 explainability panel's rationale backbone.

STRUCTURAL NOTES (flagged, not silent): OrchestratorTools DI container + `tools`
param added in 4.2; case_store added in 4.3; experience_matcher added in 4.4.
Step order/names/count unchanged throughout.
=============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.core.pipeline_registry import get_active_pipeline_version
from app.schemas.parsing import ParsedJobDescription, ParsedResume
from app.schemas.scoring import (
    ConfidenceLevel,
    FeatureVector,
    GapItem,
    ScoreResult,
    SkillMatch,
)
from app.services.confidence.confidence_utils import (
    HIGH_THRESHOLD,
    MEDIUM_THRESHOLD,
    confidence_to_band,
)
from app.services.orchestration.ambiguity_rules import (
    AmbiguityFlag,
    AmbiguitySignals,
    ConfidenceBand,
    flag_ambiguity,
)

if TYPE_CHECKING:  # heavy tool types imported for annotations only
    from app.services.rag.rag_similar_case_lookup import SimilarCaseStore
    from app.services.rag.skill_matcher import SkillMatcher
    from app.services.rag.taxonomy_schemas import SkillTaxonomyEntry
    from app.services.scoring.experience_matcher import ExperienceMatcher
    from app.services.scoring.hybrid_scorer import HybridScorer

# Number of similar past cases to pull for calibration (Phase 3.4).
CALIBRATION_TOP_K = 5

# STEP 5 ensemble weights (PRD §8.2 formula:
#   final_score = w1*tfidf + w2*embedding + w3*skill_overlap + w4*experience).
# THESE WEIGHTS ARE PLACEHOLDERS — Phase 6.4 grid search will OVERWRITE them via
# calibration against ground truth. Do NOT treat as final tuned values. Sum to 1.0.
PROVISIONAL_WEIGHTS: dict[str, float] = {
    "tfidf_score": 0.25,
    "embedding_score": 0.25,
    "skill_overlap_pct": 0.30,
    "exp_match": 0.20,
}

# STEP 5: spread across the 4 weighted features beyond which we pull the final
# confidence band down one tier (a broader internal-disagreement check than
# 4.3's Rule B, which only compares hybrid-vs-skill-overlap).
# PROVISIONAL — pending Phase 5 ground-truth calibration.
EXTREME_SIGNAL_SPREAD = 0.6

_LEVEL_RANK: dict[ConfidenceLevel, int] = {
    ConfidenceLevel.LOW: 0,
    ConfidenceLevel.MEDIUM: 1,
    ConfidenceLevel.HIGH: 2,
}
_BAND_STR_TO_LEVEL: dict[ConfidenceBand, ConfidenceLevel] = {
    "low": ConfidenceLevel.LOW,
    "medium": ConfidenceLevel.MEDIUM,
    "high": ConfidenceLevel.HIGH,
}

# The orchestrator's public output IS the locked Phase 0.2 score-response schema.
OrchestrationResult = ScoreResult

ORCHESTRATION_STEP_SEQUENCE: list[tuple[str, str]] = [
    ("hybrid_scoring", "Phase 4.2 [LIVE]"),
    ("rag_skill_matching", "Phase 4.2 [LIVE]"),
    ("experience_years_matching", "Phase 4.4 [LIVE]"),
    ("ambiguity_flagging", "Phase 4.3 [LIVE]"),
    ("decision_logic", "Phase 4.4 [LIVE]"),
]


class OrchestrationConfigError(Exception):
    """Raised when required tools are not supplied for a wired step."""


class OrchestrationValidationError(Exception):
    """Raised when a tool's input/output does not match the Phase 0.2 shape."""


class OrchestrationToolError(Exception):
    """Raised when a wired tool raises — attributable to a specific step/tool."""


@dataclass
class OrchestratorTools:
    """Injected tool instances the wired steps call (dependency injection).

    Grown across phases (each addition flagged in the file header): STEP 1-2 tools
    in 4.2, case_store for STEP 4 in 4.3, experience_matcher for STEP 3 in 4.4.
    """

    hybrid_scorer: HybridScorer
    skill_matcher: SkillMatcher
    taxonomy_entries: list[SkillTaxonomyEntry]
    # Added Phase 4.3 (flagged): STEP 4 makes a live calibration_check() call.
    case_store: SimilarCaseStore
    # Added Phase 4.4 (flagged): STEP 3 experience matcher, authored in 4.4.
    experience_matcher: ExperienceMatcher


@dataclass
class _OrchestrationState:
    """Accumulator of STEP 1-4 outputs for STEP 5 (4.4) to consume.

    Each field is traceable to the tool that produced it (Design Blueprint §10.8).
    This holds data only — it does NOT define how 4.4 will consume it.
    """

    hybrid_result: ScoreResult | None = None  # STEP 1 (hybrid scorer)
    skill_overlap_pct: float | None = None  # STEP 2 (RAG skill matcher)
    matched_skills: list[SkillMatch] = field(default_factory=list)  # STEP 2
    gaps: list[GapItem] = field(default_factory=list)  # STEP 2
    exp_match: float | None = None  # STEP 3 (experience matcher, Phase 4.4)
    ambiguity_flag: AmbiguityFlag | None = None  # STEP 4 (Phase 4.3)


def _require_parsed(resume: ParsedResume, jd: ParsedJobDescription) -> None:
    """Boundary input validation against the Phase 0.2 schema shapes."""
    if not isinstance(resume, ParsedResume):
        raise OrchestrationValidationError(
            f"expected ParsedResume input, got {type(resume).__name__}"
        )
    if not isinstance(jd, ParsedJobDescription):
        raise OrchestrationValidationError(
            f"expected ParsedJobDescription input, got {type(jd).__name__}"
        )


def _step_hybrid_scoring(
    resume: ParsedResume, jd: ParsedJobDescription, tools: OrchestratorTools
) -> ScoreResult:
    """STEP 1 (LIVE) — call the Phase 2.4 hybrid scorer; validate the ScoreResult."""
    _require_parsed(resume, jd)
    try:
        result = tools.hybrid_scorer.compute_hybrid_score(
            resume.document_id, resume, jd.document_id, jd
        )
    except Exception as exc:  # attributable, never silently absorbed
        raise OrchestrationToolError(
            f"STEP 1 hybrid_scoring: hybrid scorer raised on this input: {exc}"
        ) from exc
    if not isinstance(result, ScoreResult):
        raise OrchestrationValidationError(
            f"STEP 1 hybrid_scoring: expected ScoreResult output, "
            f"got {type(result).__name__}"
        )
    return result


def _step_rag_skill_matching(
    resume: ParsedResume, jd: ParsedJobDescription, tools: OrchestratorTools
) -> tuple[float, list[SkillMatch], list[GapItem]]:
    """STEP 2 (LIVE) — call the Phase 3.3 RAG skill matcher; validate the shape."""
    _require_parsed(resume, jd)
    try:
        out = tools.skill_matcher.match_resume_to_jd(
            resume.skills,
            jd.required_skills,
            jd.preferred_skills,
            tools.taxonomy_entries,
        )
    except Exception as exc:
        raise OrchestrationToolError(
            f"STEP 2 rag_skill_matching: skill matcher raised on this input: {exc}"
        ) from exc

    if not (isinstance(out, tuple) and len(out) == 3):
        raise OrchestrationValidationError(
            "STEP 2 rag_skill_matching: expected (float, list[SkillMatch], "
            f"list[GapItem]) output, got {type(out).__name__}"
        )
    overlap, matches, gaps = out
    if not isinstance(overlap, int | float) or not (0.0 <= float(overlap) <= 1.0):
        raise OrchestrationValidationError(
            f"STEP 2 rag_skill_matching: skill_overlap_pct out of [0,1]: {overlap!r}"
        )
    if not all(isinstance(m, SkillMatch) for m in matches):
        raise OrchestrationValidationError(
            "STEP 2 rag_skill_matching: matched_skills must be list[SkillMatch]"
        )
    if not all(isinstance(g, GapItem) for g in gaps):
        raise OrchestrationValidationError(
            "STEP 2 rag_skill_matching: gaps must be list[GapItem]"
        )
    return float(overlap), list(matches), list(gaps)


def _step_experience_years_matching(
    resume: ParsedResume, jd: ParsedJobDescription, tools: OrchestratorTools
) -> float:
    """STEP 3 (LIVE, Phase 4.4) — experience/years match → exp_match in [0,1].

    The matcher module was authored in Phase 4.4 to unblock this step (flagged).
    """
    _require_parsed(resume, jd)
    try:
        exp = tools.experience_matcher.match(resume, jd)
    except Exception as exc:
        raise OrchestrationToolError(
            f"STEP 3 experience_years_matching: matcher raised on this input: {exc}"
        ) from exc
    if not isinstance(exp, int | float) or not (0.0 <= float(exp) <= 1.0):
        raise OrchestrationValidationError(
            f"STEP 3 experience_years_matching: exp_match out of [0,1]: {exp!r}"
        )
    return float(exp)


_CALIBRATION_FIELDS = (
    "is_outlier",
    "deviation",
    "low_sample_warning",
    "similar_case_ids",
    "similar_case_scores",
)


def _step_ambiguity_flagging(
    resume: ParsedResume,
    jd: ParsedJobDescription,
    tools: OrchestratorTools,
    state: _OrchestrationState,
) -> AmbiguityFlag:
    """STEP 4 (LIVE, Phase 4.3) — run the ambiguity flagger.

    First live invocation of the Phase 3.4 similar-case calibration in the
    pipeline: retrieve similar cases, run calibration_check, validate its shape at
    this new integration boundary, then apply the deterministic ambiguity rules.
    Advisory only — Phase 4.4 owns the final decision.
    """
    _require_parsed(resume, jd)
    if state.hybrid_result is None:
        raise OrchestrationValidationError(
            "STEP 4 ambiguity_flagging: STEP 1 hybrid_result missing from state."
        )

    try:
        case_embedding = tools.case_store.build_case_embedding(
            resume.raw_text, jd.raw_text
        )
        similar = tools.case_store.retrieve_similar_cases(
            case_embedding, k=CALIBRATION_TOP_K
        )
        calibration = tools.case_store.calibration_check(
            state.hybrid_result.final_score, similar, requested_k=CALIBRATION_TOP_K
        )
    except Exception as exc:
        raise OrchestrationToolError(
            f"STEP 4 ambiguity_flagging: calibration lookup raised on this input: {exc}"
        ) from exc

    missing = [f for f in _CALIBRATION_FIELDS if not hasattr(calibration, f)]
    if missing:
        raise OrchestrationValidationError(
            f"STEP 4 ambiguity_flagging: calibration_result missing field(s) {missing}."
        )

    signals = AmbiguitySignals(
        hybrid_final_score=state.hybrid_result.final_score,
        skill_overlap_pct=state.skill_overlap_pct or 0.0,
    )
    return flag_ambiguity(signals, resume.parsing_confidence, calibration)


def _round_half_up_100(value_0_1: float) -> int:
    """Scale [0,1] → 0-100 int with round-HALF-UP (UX legibility over banker's)."""
    return int(math.floor(value_0_1 * 100 + 0.5))


def _more_restrictive_level(a: ConfidenceLevel, b: ConfidenceLevel) -> ConfidenceLevel:
    return a if _LEVEL_RANK[a] <= _LEVEL_RANK[b] else b


def _pull_down_one_level(level: ConfidenceLevel) -> ConfidenceLevel:
    if level is ConfidenceLevel.HIGH:
        return ConfidenceLevel.MEDIUM
    if level is ConfidenceLevel.MEDIUM:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.LOW


def _clamp_confidence_to_band(value: float, band: ConfidenceLevel) -> float:
    """Clamp a numeric confidence into its band's range so the number and the
    categorical band can never contradict each other."""
    if band is ConfidenceLevel.HIGH:
        return max(HIGH_THRESHOLD, min(value, 1.0))
    if band is ConfidenceLevel.MEDIUM:
        return max(MEDIUM_THRESHOLD, min(value, HIGH_THRESHOLD - 0.01))
    return max(0.0, min(value, MEDIUM_THRESHOLD - 0.01))


def compute_final_decision(
    step_outputs: _OrchestrationState, ambiguity_flag: AmbiguityFlag
) -> OrchestrationResult:
    """STEP 5 — produce the final, authoritative OrchestrationResult (ScoreResult).

    Inputs: the accumulated STEP 1-3 outputs (state) and STEP 4's advisory
    AmbiguityFlag. Output: a Phase 0.2-schema-conformant ScoreResult.

    BAND RULE (one sentence): the final band is the MORE RESTRICTIVE of the
    ambiguity flag's advisory band and the signal-agreement band, pulled down one
    further tier when the four weighted features disagree beyond EXTREME_SIGNAL_
    SPREAD — it can never be raised above the advisory band.

    Schema fields populated: final_score, feature_vector (all 5 petals),
    scoring_confidence, confidence_level (the routing signal for §6.3), parsing_
    confidence (carried), matched_skills + gaps (STEP 2 explainability),
    pipeline_version (carried unaltered), feature_importance=None (Phase 6).
    """
    hybrid = step_outputs.hybrid_result
    if hybrid is None:
        # An upstream failure must never be absorbed into a fabricated score.
        raise OrchestrationValidationError(
            "STEP 5 decision_logic: STEP 1 hybrid_result missing — upstream failure "
            "not absorbed into a default score."
        )

    tfidf = hybrid.feature_vector.tfidf_score
    embedding = hybrid.feature_vector.embedding_score
    skill_overlap = step_outputs.skill_overlap_pct or 0.0
    exp = step_outputs.exp_match or 0.0
    # No education matcher exists; edu_match is honestly 0.0 and is NOT weighted in
    # the §8.2 4-term formula (it remains in the vector for the Phase 6 model).
    feature_vector = FeatureVector(
        tfidf_score=tfidf,
        embedding_score=embedding,
        skill_overlap_pct=skill_overlap,
        exp_match=exp,
        edu_match=0.0,
    )

    weighted = (
        tfidf * PROVISIONAL_WEIGHTS["tfidf_score"]
        + embedding * PROVISIONAL_WEIGHTS["embedding_score"]
        + skill_overlap * PROVISIONAL_WEIGHTS["skill_overlap_pct"]
        + exp * PROVISIONAL_WEIGHTS["exp_match"]
    )
    final_score = _round_half_up_100(weighted)

    # --- Final confidence band (advisory-capped, spread-pulled) --------------
    advisory = _BAND_STR_TO_LEVEL[ambiguity_flag.recommended_confidence_band]
    weighted_features = [tfidf, embedding, skill_overlap, exp]
    spread = max(weighted_features) - min(weighted_features)
    base_conf = round(1.0 - spread, 6)
    signal_band = confidence_to_band(base_conf)
    final_band = _more_restrictive_level(advisory, signal_band)
    if spread > EXTREME_SIGNAL_SPREAD:
        final_band = _pull_down_one_level(final_band)
    scoring_confidence = _clamp_confidence_to_band(base_conf, final_band)

    # --- Merge reasons (UNION, not compression) — each stays traceable --------
    # STEP 4's advisory reasons first, then any STEP 5 decision-layer reasons.
    reasons = list(ambiguity_flag.reasons)
    if final_band is advisory and advisory is not signal_band:
        reasons.append(
            f"STEP 5: confidence capped at ambiguity advisory band "
            f"'{advisory.value}' (signals alone suggested '{signal_band.value}')"
        )
    if spread > EXTREME_SIGNAL_SPREAD:
        reasons.append(
            f"STEP 5: confidence pulled down one tier — feature spread "
            f"{spread:.2f} exceeds {EXTREME_SIGNAL_SPREAD}"
        )

    return OrchestrationResult(
        resume_id=hybrid.resume_id,
        jd_id=hybrid.jd_id,
        final_score=final_score,
        feature_vector=feature_vector,
        scoring_confidence=scoring_confidence,
        confidence_level=final_band,
        parsing_confidence=hybrid.parsing_confidence,
        matched_skills=step_outputs.matched_skills,  # STEP 2 explainability
        gaps=step_outputs.gaps,  # STEP 2 explainability
        confidence_reasons=reasons,  # merged STEP 4 + STEP 5 rationale (§10.8)
        feature_importance=None,  # populated by the Phase 6 ML model
        pipeline_version=hybrid.pipeline_version,  # carried unaltered
    )


def _short_circuit_low_confidence(
    resume: ParsedResume, jd: ParsedJobDescription, pipeline_version: str
) -> OrchestrationResult:
    """Honest low-confidence result for the illustrative short-circuit rule."""
    confidence = 0.0
    return OrchestrationResult(
        resume_id=resume.document_id,
        jd_id=jd.document_id,
        final_score=0,
        feature_vector=FeatureVector(
            tfidf_score=0.0,
            embedding_score=0.0,
            skill_overlap_pct=0.0,
            exp_match=0.0,
            edu_match=0.0,
        ),
        scoring_confidence=confidence,
        confidence_level=confidence_to_band(confidence),
        parsing_confidence=resume.parsing_confidence,
        confidence_reasons=[
            "short-circuit: no skills extracted from resume — cannot score "
            "meaningfully, returning honest low-confidence result"
        ],
        pipeline_version=pipeline_version,
    )


def run_orchestration(
    parsed_resume: ParsedResume,
    parsed_jd: ParsedJobDescription,
    tools: OrchestratorTools | None = None,
    pipeline_version: str | None = None,
) -> OrchestrationResult:
    """Deterministic top-level orchestration.

    STEP 1 (hybrid scoring) and STEP 2 (RAG skill matching) are LIVE and require
    ``tools``. STEP 3 (experience matcher) is BLOCKED (module missing); STEP 4/5
    remain placeholders for Phases 4.3/4.4. ``pipeline_version`` is CARRIED (not
    decided), defaulting to the active registry version (Phase 0.3).
    """
    version = pipeline_version or get_active_pipeline_version().value

    # --- ILLUSTRATIVE PATTERN (unchanged from 4.1; NOT a final rule) ---------
    if len(parsed_resume.skills) == 0:
        return _short_circuit_low_confidence(parsed_resume, parsed_jd, version)

    if tools is None:
        raise OrchestrationConfigError(
            "run_orchestration requires `tools` for the wired STEP 1-2 calls."
        )

    state = _OrchestrationState()

    # STEP 1 — hybrid scoring (LIVE)
    state.hybrid_result = _step_hybrid_scoring(parsed_resume, parsed_jd, tools)

    # STEP 2 — RAG skill matching (LIVE)
    overlap, matches, gaps = _step_rag_skill_matching(parsed_resume, parsed_jd, tools)
    state.skill_overlap_pct = overlap
    state.matched_skills = matches
    state.gaps = gaps

    # STEP 3 — experience/years matching (Phase 4.4, LIVE)
    state.exp_match = _step_experience_years_matching(parsed_resume, parsed_jd, tools)

    # STEP 4 — ambiguity flagging (Phase 4.3, LIVE)
    state.ambiguity_flag = _step_ambiguity_flagging(
        parsed_resume, parsed_jd, tools, state
    )

    # STEP 5 — final decision logic (Phase 4.4, LIVE)
    return compute_final_decision(state, state.ambiguity_flag)

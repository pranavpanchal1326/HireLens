"""Agent Orchestrator — deterministic, rule-based control-flow skeleton (Phase 4.1).

=====================================================================================
ARCHITECTURAL CONSTRAINT (NON-NEGOTIABLE, PRD §4 / §15):
This orchestrator is a DETERMINISTIC, RULE-BASED control-flow component. It is NOT
an LLM agent. There are, and must remain, ZERO LLM calls, prompts, chain-of-thought,
or natural-language decision steps anywhere in this control flow. Every routing
choice is plain, auditable if/else / rule-table code a human can trace line by line.
The only LLM permitted in the whole system is the OPTIONAL downstream polish layer
(PRD §4), which is out of scope here and does not appear in this file.
=====================================================================================

=============================== TRACEABILITY ================================
UPSTREAM (fixed — imported/carried, never redefined here):
  - Data contracts (ParsedResume, ParsedJobDescription, FeatureVector,
    ScoreResult) ................................. Phase 0.2
  - Pipeline versioning (locked enum + registry) . Phase 0.3
  - Confidence banding utility ................... Phase 1.3
DOWNSTREAM (placeholders this skeleton exposes, filled by later prompts):
  - STEP 1-3 tool wiring (hybrid scorer, RAG skill matcher, experience matcher)
    ............................................... Phase 4.2
  - STEP 4 ambiguity flagger logic ............... Phase 4.3
  - STEP 5 decision-logic / weighting / confidence  Phase 4.4
LOCALLY OWNED: the control-flow shape only (this file).
=============================================================================
"""

from __future__ import annotations

from app.core.pipeline_registry import get_active_pipeline_version
from app.schemas.parsing import ParsedJobDescription, ParsedResume
from app.schemas.scoring import FeatureVector, ScoreResult
from app.services.confidence.confidence_utils import confidence_to_band

# The orchestrator's public output IS the locked Phase 0.2 score-response schema.
# We deliberately do NOT invent a new schema — OrchestrationResult is that contract.
OrchestrationResult = ScoreResult

# The control flow, as a human-readable ordered manifest. Phase 4.2/4.3/4.4 wire
# real calls into the correspondingly-named steps in run_orchestration below. This
# list is the single, one-screen source of truth for the pipeline's shape.
ORCHESTRATION_STEP_SEQUENCE: list[tuple[str, str]] = [
    ("hybrid_scoring", "Phase 4.2"),
    ("rag_skill_matching", "Phase 4.2"),
    ("experience_years_matching", "Phase 4.2"),
    ("ambiguity_flagging", "Phase 4.3"),
    ("decision_logic", "Phase 4.4"),
]


def _not_yet(step_name: str, phase: str) -> None:
    """Fail loudly at an unwired step — never return fake data."""
    raise NotImplementedError(f"Step '{step_name}' is wired in {phase}.")


def _step_hybrid_scoring(resume: ParsedResume, jd: ParsedJobDescription) -> None:
    """STEP 1 — call the Phase 2 hybrid scorer. Placeholder until Phase 4.2."""
    _not_yet("hybrid_scoring", "Phase 4.2")


def _step_rag_skill_matching(resume: ParsedResume, jd: ParsedJobDescription) -> None:
    """STEP 2 — call the Phase 3.1-3.3 RAG skill matcher. Placeholder until 4.2."""
    _not_yet("rag_skill_matching", "Phase 4.2")


def _step_experience_years_matching(
    resume: ParsedResume, jd: ParsedJobDescription
) -> None:
    """STEP 3 — call the experience/years matcher. Placeholder until Phase 4.2."""
    _not_yet("experience_years_matching", "Phase 4.2")


def _step_ambiguity_flagging(resume: ParsedResume, jd: ParsedJobDescription) -> None:
    """STEP 4 — call the ambiguity flagger. Placeholder until Phase 4.3."""
    _not_yet("ambiguity_flagging", "Phase 4.3")


def _step_decision_logic(
    resume: ParsedResume, jd: ParsedJobDescription
) -> OrchestrationResult:
    """STEP 5 — apply decision/weighting/confidence logic. Placeholder until 4.4."""
    _not_yet("decision_logic", "Phase 4.4")
    raise AssertionError("unreachable")  # for type-checkers; _not_yet always raises


def _short_circuit_low_confidence(
    resume: ParsedResume, jd: ParsedJobDescription, pipeline_version: str
) -> OrchestrationResult:
    """Honest low-confidence result for the illustrative short-circuit rule.

    Not fabricated tool output — it is a genuine, deterministic "we can't
    meaningfully score this" result (zeroed features, LOW confidence), which is the
    honest thing to return when there is nothing to score.
    """
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
        pipeline_version=pipeline_version,
    )


def run_orchestration(
    parsed_resume: ParsedResume,
    parsed_jd: ParsedJobDescription,
    pipeline_version: str | None = None,
) -> OrchestrationResult:
    """Deterministic top-level orchestration entry point.

    Inputs: a ParsedResume and ParsedJobDescription (Phase 0.2). Output: an
    OrchestrationResult (== Phase 0.2 ScoreResult). ``pipeline_version`` is CARRIED
    (not decided) — defaults to the registry's currently-active version (Phase 0.3).

    The body is the full, auditable control flow. Real tool calls are wired into the
    STEP placeholders in Phases 4.2-4.4; until then each raises NotImplementedError
    with its owning phase rather than returning fake data.
    """
    version = pipeline_version or get_active_pipeline_version().value

    # --- ILLUSTRATIVE PATTERN (NOT a final business rule) --------------------
    # Establishes HOW deterministic short-circuit branching looks in this codebase.
    # Real rules are defined in Phase 4.3/4.4. Rule: a resume with zero extracted
    # skills cannot be meaningfully scored, so short-circuit to a low-confidence
    # result and skip the remaining steps.
    if len(parsed_resume.skills) == 0:
        return _short_circuit_low_confidence(parsed_resume, parsed_jd, version)

    # --- Main deterministic pipeline (placeholders until wired) ---------------
    _step_hybrid_scoring(parsed_resume, parsed_jd)  # STEP 1 — Phase 4.2
    _step_rag_skill_matching(parsed_resume, parsed_jd)  # STEP 2 — Phase 4.2
    _step_experience_years_matching(parsed_resume, parsed_jd)  # STEP 3 — Phase 4.2
    _step_ambiguity_flagging(parsed_resume, parsed_jd)  # STEP 4 — Phase 4.3
    return _step_decision_logic(parsed_resume, parsed_jd)  # STEP 5 — Phase 4.4

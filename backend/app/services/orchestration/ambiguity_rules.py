"""Ambiguity Flagger rules (Phase 4.3).

Deterministic gate deciding whether a scoring pass is clean enough to proceed, or
needs a deeper check / capped confidence. This is the mechanical enforcement of
Design Blueprint P3 ("Honest over impressive") at the orchestration layer.

Lives in its own module (imported by agent_orchestrator) so the rule set stays
readable and independently testable without importing heavy tool modules.

ABSOLUTE CONSTRAINT (PRD §4/§15, inherited from Phase 4.1): NO LLM, prompt, or
natural-language reasoning anywhere. Every reason string is produced by a
deterministic template from the rule that fired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

ConfidenceBand = Literal["high", "medium", "low"]

# ============================ THRESHOLD CONSTANTS ============================
# Rule A: minimum parsing_confidence (Phase 1.3, "% of expected fields extracted")
# below which the score cannot be trusted no matter how good the other signals
# look. 0.6 sits just above Phase 1.3's MEDIUM band floor (0.5) — a resume we
# could only partially read should not yield a confident score.
# THRESHOLD NEEDS CALIBRATION AGAINST GROUND TRUTH — Phase 5.
PARSING_CONFIDENCE_MIN = 0.6

# Rule B: max allowed disagreement between the hybrid score (normalized 0-1) and
# the RAG skill-overlap signal before the case is treated as internally
# inconsistent. 0.4 = the two signals point to meaningfully different conclusions.
# THRESHOLD NEEDS CALIBRATION AGAINST GROUND TRUTH — Phase 5.
SCORE_DISAGREEMENT_DELTA = 0.4

_BAND_RANK: dict[ConfidenceBand, int] = {"low": 0, "medium": 1, "high": 2}


class _CalibrationLike(Protocol):
    """Structural type of Phase 3.4's CalibrationResult (duck-typed to avoid
    importing the heavy similar-case module here)."""

    is_outlier: bool
    deviation: float
    low_sample_warning: bool
    similar_case_ids: list[str]
    similar_case_scores: list[int]


@dataclass
class AmbiguitySignals:
    """The STEP 1-3 outputs the flagger consumes (built by the orchestrator).

    hybrid_final_score: STEP 1 hybrid scorer (Phase 2.4), 0-100.
    skill_overlap_pct: STEP 2 RAG skill matcher (Phase 3.3), 0-1.
    """

    hybrid_final_score: int
    skill_overlap_pct: float


@dataclass
class AmbiguityFlag:
    """Fully inspectable ambiguity verdict (no black box — Design Blueprint §10.8).

    requires_deeper_check: whether Phase 4.4 should handle this with extra care.
    reasons: human-readable, each traceable to the exact rule/signal that fired.
    recommended_confidence_band: ADVISORY cap only — Phase 4.4 owns the final band.
    raw_signals: the actual values that fed the decision, for the explainability panel.
    """

    requires_deeper_check: bool
    reasons: list[str]
    recommended_confidence_band: ConfidenceBand
    raw_signals: dict[str, object]


@dataclass
class _RuleResult:
    reason: str
    band_cap: ConfidenceBand | None
    requires_deeper_check: bool


def _more_restrictive(current: ConfidenceBand, cap: ConfidenceBand) -> ConfidenceBand:
    """Return whichever band is stricter (low < medium < high)."""
    return current if _BAND_RANK[current] <= _BAND_RANK[cap] else cap


def rule_a_low_parsing_confidence(parsing_confidence: float) -> _RuleResult | None:
    """Trigger: parsing_confidence (Phase 1.3) below PARSING_CONFIDENCE_MIN.
    Consequence: deeper check + band forced to 'low' regardless of other signals."""
    if parsing_confidence < PARSING_CONFIDENCE_MIN:
        return _RuleResult(
            reason=(
                f"parsing_confidence {parsing_confidence:.2f} below threshold "
                f"{PARSING_CONFIDENCE_MIN}"
            ),
            band_cap="low",
            requires_deeper_check=True,
        )
    return None


def rule_b_score_disagreement(
    hybrid_normalized: float, skill_overlap_pct: float
) -> _RuleResult | None:
    """Trigger: |hybrid score (norm) - RAG skill overlap| > SCORE_DISAGREEMENT_DELTA.
    Consequence: deeper check + band capped at 'medium'."""
    delta = abs(hybrid_normalized - skill_overlap_pct)
    if delta > SCORE_DISAGREEMENT_DELTA:
        return _RuleResult(
            reason=(
                f"hybrid score {hybrid_normalized:.2f} and skill-overlap "
                f"{skill_overlap_pct:.2f} disagree by {delta:.2f} "
                f"(> {SCORE_DISAGREEMENT_DELTA})"
            ),
            band_cap="medium",
            requires_deeper_check=True,
        )
    return None


def rule_c_calibration_outlier(calibration: _CalibrationLike) -> _RuleResult | None:
    """Trigger: Phase 3.4 calibration_check flags this score a statistical outlier.
    Consequence: deeper check + band capped at 'medium'."""
    if calibration.is_outlier:
        return _RuleResult(
            reason=(
                f"calibration flags outlier: score deviates "
                f"{calibration.deviation:.1f} from similar cases "
                f"{calibration.similar_case_ids}"
            ),
            band_cap="medium",
            requires_deeper_check=True,
        )
    return None


def rule_d_low_sample(calibration: _CalibrationLike) -> _RuleResult | None:
    """Trigger: Phase 3.4 calibration base is thin (low_sample_warning).
    Consequence: NO forced low band (a cold-start store is not itself evidence of a
    bad score), but the thin base MUST be surfaced as a reason — never let a weak
    calibration base pass silently as if robust (PRD §7.3/§14 honesty requirement)."""
    if calibration.low_sample_warning:
        return _RuleResult(
            reason=(
                f"calibration base is thin (low_sample_warning): "
                f"{len(calibration.similar_case_scores)} similar case(s) — result "
                f"not corroborated by robust precedent"
            ),
            band_cap=None,
            requires_deeper_check=False,
        )
    return None


def flag_ambiguity(
    step_1_3_outputs: AmbiguitySignals,
    parsing_confidence: float,
    calibration_result: _CalibrationLike,
) -> AmbiguityFlag:
    """Aggregate rules A-D into an advisory AmbiguityFlag.

    Inputs: STEP 1-3 signals (this file's AmbiguitySignals), parsing_confidence
    (Phase 1.3), and a CalibrationResult (Phase 3.4). Output: an AmbiguityFlag whose
    recommended_confidence_band is the MOST RESTRICTIVE cap any triggered rule
    imposes. Advisory only — Phase 4.4 makes the final decision.
    """
    hybrid_normalized = step_1_3_outputs.hybrid_final_score / 100.0
    outcomes = [
        rule_a_low_parsing_confidence(parsing_confidence),
        rule_b_score_disagreement(
            hybrid_normalized, step_1_3_outputs.skill_overlap_pct
        ),
        rule_c_calibration_outlier(calibration_result),
        rule_d_low_sample(calibration_result),
    ]

    reasons: list[str] = []
    requires_deeper_check = False
    band: ConfidenceBand = "high"
    for outcome in outcomes:
        if outcome is None:
            continue
        reasons.append(outcome.reason)
        if outcome.requires_deeper_check:
            requires_deeper_check = True
        if outcome.band_cap is not None:
            band = _more_restrictive(band, outcome.band_cap)

    raw_signals: dict[str, object] = {
        "parsing_confidence": parsing_confidence,
        "hybrid_final_score": step_1_3_outputs.hybrid_final_score,
        "hybrid_normalized": round(hybrid_normalized, 4),
        "skill_overlap_pct": step_1_3_outputs.skill_overlap_pct,
        "score_disagreement": round(
            abs(hybrid_normalized - step_1_3_outputs.skill_overlap_pct), 4
        ),
        "is_outlier": calibration_result.is_outlier,
        "deviation": calibration_result.deviation,
        "low_sample_warning": calibration_result.low_sample_warning,
        "similar_case_ids": list(calibration_result.similar_case_ids),
        "similar_case_scores": list(calibration_result.similar_case_scores),
    }
    return AmbiguityFlag(
        requires_deeper_check=requires_deeper_check,
        reasons=reasons,
        recommended_confidence_band=band,
        raw_signals=raw_signals,
    )

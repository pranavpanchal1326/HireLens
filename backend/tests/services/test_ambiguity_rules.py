"""Tests for the Phase 4.3 ambiguity flagger rules."""

from __future__ import annotations

from dataclasses import dataclass

from app.services.orchestration.ambiguity_rules import (
    PARSING_CONFIDENCE_MIN,
    SCORE_DISAGREEMENT_DELTA,
    AmbiguitySignals,
    flag_ambiguity,
)


@dataclass
class _Calibration:
    """Stub matching Phase 3.4 CalibrationResult's structural shape."""

    is_outlier: bool = False
    deviation: float = 0.0
    low_sample_warning: bool = False
    similar_case_ids: list[str] | None = None
    similar_case_scores: list[int] | None = None

    def __post_init__(self) -> None:
        if self.similar_case_ids is None:
            self.similar_case_ids = []
        if self.similar_case_scores is None:
            self.similar_case_scores = []


def _clean_cal() -> _Calibration:
    # Enough sample, not an outlier, not thin.
    return _Calibration(
        is_outlier=False,
        deviation=2.0,
        low_sample_warning=False,
        similar_case_ids=["c1", "c2", "c3", "c4", "c5"],
        similar_case_scores=[80, 78, 82, 79, 81],
    )


def _signals(final: int, overlap: float) -> AmbiguitySignals:
    return AmbiguitySignals(hybrid_final_score=final, skill_overlap_pct=overlap)


def test_happy_path_no_rule_triggers() -> None:
    flag = flag_ambiguity(_signals(80, 0.78), 0.9, _clean_cal())
    assert flag.requires_deeper_check is False
    assert flag.reasons == []
    assert flag.recommended_confidence_band == "high"


def test_rule_a_low_parsing_confidence_forces_low() -> None:
    flag = flag_ambiguity(_signals(80, 0.78), 0.42, _clean_cal())
    assert flag.requires_deeper_check is True
    assert flag.recommended_confidence_band == "low"
    assert any("parsing_confidence" in r for r in flag.reasons)


def test_rule_b_score_disagreement_caps_medium() -> None:
    # hybrid 0.90 vs skill overlap 0.10 → delta 0.80 > threshold.
    flag = flag_ambiguity(_signals(90, 0.10), 0.9, _clean_cal())
    assert flag.requires_deeper_check is True
    assert flag.recommended_confidence_band == "medium"
    assert any("disagree" in r for r in flag.reasons)


def test_rule_c_outlier_caps_medium() -> None:
    cal = _clean_cal()
    cal.is_outlier = True
    cal.deviation = 40.0
    flag = flag_ambiguity(_signals(80, 0.78), 0.9, cal)
    assert flag.requires_deeper_check is True
    assert flag.recommended_confidence_band == "medium"
    assert any("outlier" in r for r in flag.reasons)


def test_rule_d_low_sample_surfaces_but_does_not_force_low() -> None:
    cal = _Calibration(low_sample_warning=True, similar_case_scores=[80])
    flag = flag_ambiguity(_signals(80, 0.78), 0.9, cal)
    # Surfaced as a reason...
    assert any("thin" in r or "low_sample" in r for r in flag.reasons)
    # ...but does NOT force low band or a deeper check on its own.
    assert flag.recommended_confidence_band == "high"
    assert flag.requires_deeper_check is False


def test_multiple_rules_most_restrictive_band_wins() -> None:
    # Rule A (low parsing) → 'low' must win over Rule B/C 'medium'.
    cal = _clean_cal()
    cal.is_outlier = True
    flag = flag_ambiguity(_signals(90, 0.10), 0.40, cal)
    assert flag.recommended_confidence_band == "low"  # A beats B and C
    assert flag.requires_deeper_check is True
    # All three reasons present.
    assert len(flag.reasons) >= 3


def test_raw_signals_are_fully_inspectable() -> None:
    flag = flag_ambiguity(_signals(90, 0.10), 0.42, _clean_cal())
    rs = flag.raw_signals
    for key in (
        "parsing_confidence",
        "hybrid_final_score",
        "skill_overlap_pct",
        "score_disagreement",
        "is_outlier",
        "low_sample_warning",
    ):
        assert key in rs
    assert rs["parsing_confidence"] == 0.42


def test_threshold_boundaries_are_exclusive_as_documented() -> None:
    # Exactly at PARSING_CONFIDENCE_MIN → NOT triggered (strict <).
    at_threshold = flag_ambiguity(
        _signals(80, 0.78), PARSING_CONFIDENCE_MIN, _clean_cal()
    )
    assert at_threshold.recommended_confidence_band == "high"
    # Disagreement exactly at delta → NOT triggered (strict >).
    overlap = 0.80 - SCORE_DISAGREEMENT_DELTA  # delta == threshold exactly
    at_delta = flag_ambiguity(_signals(80, overlap), 0.9, _clean_cal())
    assert at_delta.recommended_confidence_band == "high"

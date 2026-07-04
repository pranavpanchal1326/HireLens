"""Tests for the Phase 4.2 orchestrator tool wiring (STEP 1-2 live)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from app.schemas.parsing import ParsedJobDescription, ParsedResume
from app.schemas.scoring import (
    ConfidenceLevel,
    FeatureVector,
    GapItem,
    ScoreResult,
    SkillMatch,
)
from app.services.orchestration.agent_orchestrator import (
    OrchestrationResult,
    OrchestrationToolError,
    OrchestrationValidationError,
    OrchestratorTools,
    _OrchestrationState,
    _step_ambiguity_flagging,
    _step_hybrid_scoring,
    _step_rag_skill_matching,
    compute_final_decision,
    run_orchestration,
)
from app.services.orchestration.ambiguity_rules import AmbiguityFlag
from app.services.scoring.experience_matcher import ExperienceMatcher


def _resume(skills: list[str]) -> ParsedResume:
    return ParsedResume(
        raw_text="x",
        skills=skills,
        experience=[],
        education=[],
        total_years_experience=None,
        contact_info_present=False,
        parsing_confidence=0.7,
        parsing_warnings=[],
        pipeline_version="parser-v1",
    )


def _jd() -> ParsedJobDescription:
    return ParsedJobDescription(
        raw_text="x",
        required_skills=["Python"],
        preferred_skills=[],
        required_years_experience=None,
        required_education_level=None,
        parsing_confidence=0.8,
        pipeline_version="parser-v1",
    )


def _score_result() -> ScoreResult:
    return ScoreResult(
        resume_id="r",
        jd_id="j",
        final_score=60,
        feature_vector=FeatureVector(
            tfidf_score=0.5,
            embedding_score=0.6,
            skill_overlap_pct=0.0,
            exp_match=0.0,
            edu_match=0.0,
        ),
        scoring_confidence=0.5,
        confidence_level=ConfidenceLevel.MEDIUM,
        parsing_confidence=0.7,
        pipeline_version="v3-hybrid",
    )


class _StubHybrid:
    def __init__(self, ret: object) -> None:
        self._ret = ret

    def compute_hybrid_score(self, *_args: object, **_kw: object) -> object:
        if isinstance(self._ret, Exception):
            raise self._ret
        return self._ret


class _StubMatcher:
    def __init__(self, ret: object) -> None:
        self._ret = ret

    def match_resume_to_jd(self, *_args: object, **_kw: object) -> object:
        if isinstance(self._ret, Exception):
            raise self._ret
        return self._ret


@dataclass
class _StubCalibration:
    is_outlier: bool = False
    deviation: float = 0.0
    low_sample_warning: bool = True
    similar_case_ids: list[str] = field(default_factory=list)
    similar_case_scores: list[int] = field(default_factory=list)


class _StubCaseStore:
    """Minimal Phase 3.4 store stand-in for STEP 4 wiring tests."""

    def __init__(self, calibration: object | Exception) -> None:
        self._calibration = calibration

    def build_case_embedding(self, *_a: object, **_k: object) -> object:
        return object()

    def retrieve_similar_cases(self, *_a: object, **_k: object) -> list[object]:
        return []

    def calibration_check(self, *_a: object, **_k: object) -> object:
        if isinstance(self._calibration, Exception):
            raise self._calibration
        return self._calibration


def _tools(
    hybrid_ret: object,
    matcher_ret: object,
    calibration: object | Exception | None = None,
) -> OrchestratorTools:
    return OrchestratorTools(
        hybrid_scorer=_StubHybrid(hybrid_ret),  # type: ignore[arg-type]
        skill_matcher=_StubMatcher(matcher_ret),  # type: ignore[arg-type]
        taxonomy_entries=[],
        case_store=_StubCaseStore(calibration or _StubCalibration()),  # type: ignore[arg-type]
        experience_matcher=ExperienceMatcher(),
    )


# --- happy path (STEP 1-2 wired via stubs) -----------------------------------


def test_step1_and_step2_wire_and_validate() -> None:
    tools = _tools(
        _score_result(),
        (
            0.5,
            [
                SkillMatch(
                    resume_skill="a",
                    jd_skill="a",
                    match_type="exact",
                    similarity_score=1.0,
                )
            ],
            [
                GapItem(
                    missing_skill="b",
                    suggested_action="Highlight any experience with b.",
                )
            ],
        ),
    )
    r = _resume(["Python"])
    assert isinstance(_step_hybrid_scoring(r, _jd(), tools), ScoreResult)
    overlap, matches, gaps = _step_rag_skill_matching(r, _jd(), tools)
    assert overlap == 0.5 and len(matches) == 1 and len(gaps) == 1


def test_end_to_end_runs_with_zero_placeholders() -> None:
    # Milestone: run_orchestration completes STEP 1-5 and returns a ScoreResult.
    tools = _tools(_score_result(), (0.55, [], []))
    result = run_orchestration(_resume(["Python"]), _jd(), tools)
    assert isinstance(result, ScoreResult)
    assert 0 <= result.final_score <= 100
    assert result.confidence_level in ConfidenceLevel
    assert result.pipeline_version == "v3-hybrid"  # carried unaltered from STEP 1
    # Stub calibration flags low_sample → that rationale reaches the final result.
    assert any("thin" in r or "low_sample" in r for r in result.confidence_reasons)


def test_missing_tools_raises_config_error_not_fake_result() -> None:
    from app.services.orchestration.agent_orchestrator import OrchestrationConfigError

    with pytest.raises(OrchestrationConfigError):
        run_orchestration(_resume(["Python"]), _jd())  # skills present, no tools


# --- schema validation failures at each boundary -----------------------------


def test_step1_output_validation_failure() -> None:
    tools = _tools("not a score result", (0.0, [], []))
    with pytest.raises(OrchestrationValidationError) as exc:
        _step_hybrid_scoring(_resume(["Python"]), _jd(), tools)
    assert "STEP 1" in str(exc.value)


def test_step2_output_shape_validation_failure() -> None:
    tools = _tools(_score_result(), "not a tuple")
    with pytest.raises(OrchestrationValidationError) as exc:
        _step_rag_skill_matching(_resume(["Python"]), _jd(), tools)
    assert "STEP 2" in str(exc.value)


def test_step2_overlap_out_of_range_validation_failure() -> None:
    tools = _tools(_score_result(), (1.5, [], []))  # overlap > 1.0
    with pytest.raises(OrchestrationValidationError) as exc:
        _step_rag_skill_matching(_resume(["Python"]), _jd(), tools)
    assert "out of [0,1]" in str(exc.value)


def test_input_validation_rejects_wrong_type() -> None:
    tools = _tools(_score_result(), (0.0, [], []))
    with pytest.raises(OrchestrationValidationError):
        _step_hybrid_scoring("not a resume", _jd(), tools)  # type: ignore[arg-type]


# --- named tool exception ----------------------------------------------------


def test_tool_exception_is_named_not_swallowed() -> None:
    tools = _tools(ValueError("boom"), (0.0, [], []))
    with pytest.raises(OrchestrationToolError) as exc:
        _step_hybrid_scoring(_resume(["Python"]), _jd(), tools)
    assert "hybrid scorer raised" in str(exc.value)


# --- STEP 4 wiring (Phase 4.3, LIVE) -----------------------------------------


def _state_with_hybrid() -> _OrchestrationState:
    state = _OrchestrationState()
    state.hybrid_result = _score_result()
    state.skill_overlap_pct = 0.55
    return state


def test_step4_produces_ambiguity_flag() -> None:
    tools = _tools(_score_result(), (0.0, [], []), _StubCalibration())
    flag = _step_ambiguity_flagging(
        _resume(["Python"]), _jd(), tools, _state_with_hybrid()
    )
    assert isinstance(flag, AmbiguityFlag)
    # Stub calibration has low_sample_warning=True → surfaced as a reason.
    assert any("thin" in r or "low_sample" in r for r in flag.reasons)


def test_step4_validates_calibration_shape() -> None:
    # calibration_check returns an object missing required fields.
    tools = _tools(_score_result(), (0.0, [], []), object())
    with pytest.raises(OrchestrationValidationError) as exc:
        _step_ambiguity_flagging(
            _resume(["Python"]), _jd(), tools, _state_with_hybrid()
        )
    assert "STEP 4" in str(exc.value)


def test_step4_names_calibration_tool_exception() -> None:
    tools = _tools(_score_result(), (0.0, [], []), RuntimeError("cal boom"))
    with pytest.raises(OrchestrationToolError) as exc:
        _step_ambiguity_flagging(
            _resume(["Python"]), _jd(), tools, _state_with_hybrid()
        )
    assert "STEP 4" in str(exc.value)


# --- STEP 5 decision logic (Phase 4.4) ---------------------------------------


def _state(overlap: float, exp: float, matches=None, gaps=None) -> _OrchestrationState:
    state = _OrchestrationState()
    state.hybrid_result = _score_result()
    state.skill_overlap_pct = overlap
    state.exp_match = exp
    state.matched_skills = matches or []
    state.gaps = gaps or []
    return state


def _flag(band: str, reasons=None) -> AmbiguityFlag:
    return AmbiguityFlag(
        requires_deeper_check=band != "high",
        reasons=reasons or [],
        recommended_confidence_band=band,  # type: ignore[arg-type]
        raw_signals={},
    )


def test_final_score_uses_provisional_weighted_formula() -> None:
    # hybrid fv: tfidf 0.5, embedding 0.6 (from _score_result). overlap 0.5, exp 0.5.
    # 0.25*0.5 + 0.25*0.6 + 0.30*0.5 + 0.20*0.5 = 0.125+0.15+0.15+0.10 = 0.525 → 53.
    result = compute_final_decision(_state(0.5, 0.5), _flag("high"))
    assert result.final_score == 53
    fv = result.feature_vector
    assert fv.tfidf_score == 0.5 and fv.exp_match == 0.5 and fv.edu_match == 0.0


def test_step5_cannot_raise_band_above_advisory() -> None:
    # Even with perfectly-agreeing signals, an advisory of 'low' caps the result.
    result = compute_final_decision(_state(0.5, 0.5), _flag("low"))
    assert result.confidence_level is ConfidenceLevel.LOW
    # numeric confidence is clamped into the low band → never contradicts it.
    assert result.scoring_confidence < 0.5


def test_step5_spread_pulls_band_below_advisory() -> None:
    # Advisory 'high', but extreme feature spread (overlap 1.0 vs exp 0.0) pulls down.
    result = compute_final_decision(_state(1.0, 0.0), _flag("high"))
    assert result.confidence_level is not ConfidenceLevel.HIGH


def test_step5_merges_step2_explainability() -> None:
    matches = [
        SkillMatch(
            resume_skill="a", jd_skill="a", match_type="exact", similarity_score=1.0
        )
    ]
    gaps = [GapItem(missing_skill="b", suggested_action="Highlight b.")]
    result = compute_final_decision(_state(0.5, 0.5, matches, gaps), _flag("high"))
    assert result.matched_skills == matches  # not summarized away
    assert result.gaps == gaps


def test_step5_merges_confidence_reasons_union_not_compressed() -> None:
    # STEP 4 reasons must all survive into the final result, plus STEP 5's own.
    step4_reasons = ["ruleC: outlier deviation 40", "ruleD: thin calibration base"]
    # advisory 'low' caps a signals-'high' case → STEP 5 adds a capping reason.
    result = compute_final_decision(_state(0.5, 0.5), _flag("low", step4_reasons))
    for r in step4_reasons:
        assert r in result.confidence_reasons  # union: none dropped
    assert any("STEP 5" in r for r in result.confidence_reasons)  # own reason added


def test_step5_propagates_upstream_failure_not_fake_score() -> None:
    empty = _OrchestrationState()  # hybrid_result is None (upstream failed)
    with pytest.raises(OrchestrationValidationError) as exc:
        compute_final_decision(empty, _flag("high"))
    assert "upstream failure" in str(exc.value)


def test_step5_output_conforms_to_score_schema() -> None:
    result = compute_final_decision(_state(0.5, 0.5), _flag("medium"))
    # model_dump round-trips → schema-conformant, no missing/extra fields.
    dumped = result.model_dump()
    assert set(dumped.keys()) == {
        "score_id",
        "resume_id",
        "jd_id",
        "final_score",
        "feature_vector",
        "scoring_confidence",
        "confidence_level",
        "parsing_confidence",
        "matched_skills",
        "gaps",
        "confidence_reasons",
        "feature_importance",
        "pipeline_version",
        "created_at",
    }


def test_experience_matcher_deterministic_rules() -> None:
    m = ExperienceMatcher()
    r5 = _resume(["Python"]).model_copy(update={"total_years_experience": 5.0})
    r2 = _resume(["Python"]).model_copy(update={"total_years_experience": 2.0})
    jd5 = _jd().model_copy(update={"required_years_experience": 5.0})
    jd_none = _jd()  # no requirement
    assert m.match(r5, jd5) == 1.0  # meets requirement
    assert m.match(r2, jd5) == 0.4  # 2/5
    assert m.match(r5, jd_none) == 1.0  # nothing to miss
    r_unknown = _resume(["Python"])  # total_years None
    assert m.match(r_unknown, jd5) == 0.0  # requirement not demonstrated


def test_short_circuit_preserved_and_needs_no_tools() -> None:
    result = run_orchestration(_resume([]), _jd())  # zero skills, no tools
    assert isinstance(result, OrchestrationResult)
    assert result.final_score == 0
    assert result.confidence_level is ConfidenceLevel.LOW

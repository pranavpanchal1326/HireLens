"""Experience/Years Matcher (built in Phase 4.4 to unblock orchestrator STEP 3).

Deterministic comparison of a resume's total years of experience against a JD's
required years, producing the ``exp_match`` feature in [0,1] for the FeatureVector
(Phase 0.2) and the §8.2 weighted formula.

FLAGGED: this module did not exist before Phase 4.4. It was authored here (openly,
not by fabricating an external interface) because run_orchestration cannot complete
end-to-end without a STEP 3 output. Ideally it would have been its own micro-phase.
"""

from __future__ import annotations

from app.schemas.parsing import ParsedJobDescription, ParsedResume

_MATCH_PRECISION = 6


class ExperienceMatcher:
    """Computes exp_match from parsed years. Pure/deterministic — no state."""

    def match(self, resume: ParsedResume, jd: ParsedJobDescription) -> float:
        """Return exp_match in [0,1].

        Rules (each deliberate, all provisional pending Phase 5 validation):
          - JD states no requirement (None or <= 0)  -> 1.0 (nothing to miss).
          - JD requires years but resume years are unknown (None) -> 0.0 (the
            requirement is not demonstrably met; the SEPARATE parsing_confidence
            signal already records that this may be a parse gap vs a real gap).
          - Otherwise -> min(resume_years / required_years, 1.0), so meeting or
            exceeding the requirement scores 1.0 and partial experience scales
            linearly.
        """
        required = jd.required_years_experience
        if required is None or required <= 0:
            return 1.0
        resume_years = resume.total_years_experience
        if resume_years is None:
            return 0.0
        return round(min(resume_years / required, 1.0), _MATCH_PRECISION)

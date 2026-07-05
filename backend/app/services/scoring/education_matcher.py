"""Education matcher — the ``edu_match`` feature (Phase R6).

Produces the 5th feature-vector petal (PRD §5 / Design Blueprint §6.2). Per the R6
decision, edu_match is LIVE (so the aperture bloom's 5th petal is real) but is NOT
weighted into the §8.2 4-term final_score — it exists for the feature vector /
Phase-6 model / petal rendering only.

BIAS SAFETY (PRD §7.4, non-negotiable): this matcher looks ONLY at degree LEVEL
(high-school → associate → bachelor → master → doctorate). It deliberately ignores
``institution`` and ``field_of_study`` — matching on university name would
reintroduce exactly the prestige/identity bias the name-swap harness exists to
catch. Comparing degree level is a defensible, identity-neutral signal.

Career-switcher friendliness (§8.2 names this edge case): under-qualifying on
degree level scores PROPORTIONALLY, never a hard 0 gate — a strong candidate one
level short is not zeroed out.

What this module does NOT do: it does not parse resumes (consumes the parser's
structured ``education`` list), does not touch scoring weights, and does not enter
the weighted final_score.
"""

from __future__ import annotations

from app.schemas.parsing import ParsedJobDescription, ParsedResume

_MATCH_PRECISION = 4

# Ordinal degree levels. Higher = more advanced. 0 = none/unrecognized.
_LEVEL_NONE = 0
_LEVEL_HIGHSCHOOL = 1
_LEVEL_ASSOCIATE = 2
_LEVEL_BACHELOR = 3
_LEVEL_MASTER = 4
_LEVEL_DOCTORATE = 5

# Keyword → level. Checked longest-first-ish via explicit ordering below; all
# lowercase, substring-matched against the degree text. Reviewable by design.
_LEVEL_KEYWORDS: list[tuple[int, tuple[str, ...]]] = [
    (_LEVEL_DOCTORATE, ("phd", "ph.d", "doctor", "doctorate", "dphil", "d.phil")),
    (_LEVEL_MASTER, ("master", "msc", "m.sc", "m.s", "mba", "m.eng", "meng", "m.a", "postgraduate")),
    (_LEVEL_BACHELOR, ("bachelor", "bsc", "b.sc", "b.s", "b.eng", "beng", "b.tech", "btech", "b.a", "undergraduate")),
    (_LEVEL_ASSOCIATE, ("associate", "a.a", "a.s", "diploma", "foundation")),
    (_LEVEL_HIGHSCHOOL, ("high school", "highschool", "secondary", "ged", "hsc", "a-level", "matriculation")),
]


def degree_to_level(text: str | None) -> int:
    """Map a free-text degree/requirement string to an ordinal level (0-5)."""
    if not text:
        return _LEVEL_NONE
    lowered = f" {text.lower().strip()} "
    for level, keywords in _LEVEL_KEYWORDS:
        if any(kw in lowered for kw in keywords):
            return level
    return _LEVEL_NONE


class EducationMatcher:
    """Stateless degree-level matcher. Safe to reuse as a module singleton."""

    def match(self, resume: ParsedResume, jd: ParsedJobDescription) -> float:
        """Return edu_match in [0.0, 1.0] from degree levels only.

        Rules:
          - No JD requirement (unparseable/absent) → 1.0 (nothing to satisfy),
            mirroring the experience matcher's no-requirement convention.
          - Resume level >= required level → 1.0.
          - Resume level < required → proportional (resume/required), never a hard
            zero unless the resume genuinely has no recognizable degree.
        """
        required_level = degree_to_level(jd.required_education_level)
        if required_level == _LEVEL_NONE:
            return 1.0

        resume_level = max(
            (degree_to_level(e.degree) for e in resume.education),
            default=_LEVEL_NONE,
        )
        if resume_level >= required_level:
            return 1.0
        return round(resume_level / required_level, _MATCH_PRECISION)

"""Shared scoring-input text assembly (Phase 2.2 refactor).

Both the TF-IDF scorer (2.1) and the embedding scorer (2.2) must feed IDENTICAL
input text into their comparisons — otherwise the two ablation stages would be
measuring subtly different inputs, silently invalidating the Phase 5 ablation
study. This single source of truth prevents that drift.
"""

from __future__ import annotations

from app.schemas.parsing import ParsedJobDescription, ParsedResume


def prepare_resume_text_for_scoring(parsed_resume: ParsedResume) -> str:
    """Assemble the resume fields most relevant to fit into one blob.

    Included (and why): skills (the primary signal), experience entry
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
    """Assemble JD fields for fit: required + preferred skills.

    These are the fields a resume is actually matched against; the surrounding
    boilerplate of a posting adds noise rather than fit signal.
    """
    parts: list[str] = [*parsed_jd.required_skills, *parsed_jd.preferred_skills]
    return " ".join(p for p in parts if p).strip()

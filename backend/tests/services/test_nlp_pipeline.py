"""Focused unit tests for the Phase 1.2 spaCy structuring layer.

Small hardcoded synthetic snippets — diverse real-resume format testing is
Part 1.4's job. These prove each unit works correctly in isolation.
"""

from __future__ import annotations

from app.schemas.parsing import ExperienceEntry, ParsingWarningCode
from app.services.structuring.nlp_pipeline import (
    EducationExtractor,
    ExperienceExtractor,
    JobDescriptionStructurer,
    SectionSegmenter,
    SkillExtractor,
    TotalExperienceCalculator,
    _detect_contact_info_present,
)

HEADERED_RESUME = """SUMMARY
Experienced engineer.

SKILLS
Python, SQL, Docker, Machine Learning

WORK EXPERIENCE
Senior Engineer, Acme Corp
Jan 2020 - Present
Built data pipelines.

EDUCATION
Bachelor of Science in Computer Science
Massachusetts Institute of Technology, 2015
"""

NO_HEADER_RESUME = """Jane Doe is a software developer.
She knows Python and JavaScript and has worked at various companies.
"""


def test_segmenter_splits_headered_resume() -> None:
    seg = SectionSegmenter()
    sections = seg.segment(HEADERED_RESUME)
    assert seg.used_fallback is False
    assert "skills" in sections
    assert "experience" in sections
    assert "education" in sections
    assert "Python" in sections["skills"]


def test_segmenter_fallback_flags_when_no_headers() -> None:
    seg = SectionSegmenter()
    sections = seg.segment(NO_HEADER_RESUME)
    assert seg.used_fallback is True
    # Text is never dropped — it lands in 'unclassified'.
    assert "unclassified" in sections
    assert "Python" in sections["unclassified"]


def test_skill_extractor_dedupes_normalizes_and_sorts() -> None:
    ext = SkillExtractor()
    text = "python, PYTHON, Python and sql plus Docker and DOCKER"
    skills = ext.extract_skills(text)
    assert skills == sorted(skills)  # deterministic ordering
    assert skills.count("Python") == 1  # dedup + canonical casing
    assert "SQL" in skills
    assert "Docker" in skills


def test_skill_extractor_flags_empty_section() -> None:
    ext = SkillExtractor()
    skills = ext.extract_skills("this text contains no known skills at all here")
    assert skills == []
    assert ParsingWarningCode.SKILL_SECTION_EMPTY_AFTER_EXTRACTION in ext.warnings


def test_experience_extractor_parses_entries_including_present() -> None:
    text = (
        "Senior Engineer, Acme Corp\n"
        "Jan 2020 - Present\n"
        "Led the platform team.\n"
        "\n"
        "Junior Developer, Globex Inc\n"
        "Jun 2017 - Dec 2019\n"
        "Wrote services.\n"
    )
    ext = ExperienceExtractor()
    entries = ext.extract_experience(text)
    assert len(entries) == 2
    present_entry = entries[0]
    assert present_entry.start_date == "2020-01"
    assert present_entry.end_date is None  # "Present"
    closed_entry = entries[1]
    assert closed_entry.start_date == "2017-06"
    assert closed_entry.end_date == "2019-12"


def test_total_experience_no_double_count_on_overlap() -> None:
    """Two overlapping 2-year ranges over a 2018-2021 span ≈ 3 years, not 5."""
    entries = [
        ExperienceEntry(description="A", start_date="2018-01", end_date="2020-01"),
        ExperienceEntry(description="B", start_date="2019-01", end_date="2021-01"),
    ]
    total = TotalExperienceCalculator().calculate_total_years(entries)
    assert total is not None
    # Merged span 2018-01 → 2021-01 = ~3 years. Naive sum would give ~4.
    assert 2.9 <= total <= 3.1


def test_total_experience_disjoint_ranges_sum() -> None:
    entries = [
        ExperienceEntry(description="A", start_date="2015-01", end_date="2016-01"),
        ExperienceEntry(description="B", start_date="2018-01", end_date="2019-01"),
    ]
    total = TotalExperienceCalculator().calculate_total_years(entries)
    assert total is not None
    assert 1.9 <= total <= 2.1


def test_education_extractor_detects_degree_and_year() -> None:
    text = "Bachelor of Science in Computer Science, Stanford University, 2016"
    entries = EducationExtractor().extract_education(text)
    assert len(entries) >= 1
    assert entries[0].degree == "Bachelor's"
    assert entries[0].graduation_year == 2016


def test_jd_structurer_separates_required_vs_preferred() -> None:
    jd = (
        "Required: Python and SQL are must have skills.\n"
        "Preferred: Docker and Kubernetes are nice to have.\n"
        "We need minimum 5 years of experience.\n"
        "A Bachelor degree is required.\n"
    )
    result = JobDescriptionStructurer().structure(jd)
    assert "Python" in result["required_skills"]
    assert "SQL" in result["required_skills"]
    assert "Docker" in result["preferred_skills"]
    assert result["required_years_experience"] == 5.0
    assert result["required_education_level"] == "Bachelor's"


def test_contact_info_detection_boolean_only() -> None:
    # We assert ONLY the boolean — never the matched PII string.
    assert _detect_contact_info_present("Reach me at jane.doe@example.com") is True
    assert _detect_contact_info_present("Call +1 415 555 0100 anytime") is True
    assert _detect_contact_info_present("No contact details in this text") is False

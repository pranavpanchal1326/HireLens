"""spaCy structuring layer (Phase 1.2).

Turns raw extracted text (from Phase 1.1's ``ExtractionResult``) into the locked
``ParsedResume`` / ``ParsedJobDescription`` shapes from Phase 0.2.

Precision-first: a smaller, accurate skill list beats a bloated noisy one; an
uncertain field is left ``None`` rather than guessed (Design Blueprint P3).

Scope boundaries: NO final parsing_confidence (Phase 1.3), NO scoring/matching/ML
(Phases 2/3/6). The seed skill vocabulary here is a temporary bootstrap,
SUPERSEDED by the ESCO taxonomy in Phase 3.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import spacy
from spacy.matcher import PhraseMatcher

from app.schemas.parsing import (
    EducationEntry,
    ExperienceEntry,
    ExtractionResult,
    ParsedJobDescription,
    ParsedResume,
    ParsingWarningCode,
)
from app.services.confidence.parsing_confidence import (
    calculate_jd_parsing_confidence,
    calculate_resume_parsing_confidence,
)

# Load the small English model ONCE at import time. Reloading a spaCy model per
# request is a common, costly mistake (hundreds of ms each) — the model is
# stateless for our read-only use, so a single shared instance is correct.
_NLP = spacy.load("en_core_web_sm", disable=["lemmatizer"])

_SEED_SKILLS_PATH = Path(__file__).resolve().parents[2] / "data" / "seed_skills.txt"

PARSER_PIPELINE_VERSION = "parser-v1"

# --- Shared degree mapping (imported by both resume + JD extractors) ----------
# Explicit mapping, not scattered `in` checks. Order matters: higher degrees are
# checked first so "Master of Science" isn't mislabeled by a "science" substring.
DEGREE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("PhD", ("ph.d", "phd", "doctorate", "doctoral")),
    ("Master's", ("master", "m.s", "msc", "m.sc", "m.a", "mba", "m.eng")),
    ("Bachelor's", ("bachelor", "b.s", "bsc", "b.sc", "b.a", "b.eng", "b.tech")),
    ("Associate's", ("associate", "a.a", "a.s")),
]

# --- Title/company classification (Phase 1.4 fix) -----------------------------
# Deterministic signals so title/company assignment is order-independent and does
# not depend on spaCy mislabeling a job title as an ORG. spaCy ORG is only a last
# resort when the suffix heuristic finds no company.
_COMPANY_SUFFIX_RE = re.compile(
    r"\b(inc|llc|ltd|co|corp|corporation|company|technologies|technology|"
    r"solutions|systems|group|labs|holdings|partners|associates|university|"
    r"college|institute|bank|ventures|studio|studios|agency|consulting|gmbh|"
    r"plc|sarl|bv|ag)\b\.?",
    re.IGNORECASE,
)
_TITLE_KEYWORD_RE = re.compile(
    r"\b(engineer|developer|manager|analyst|designer|director|consultant|"
    r"supervisor|lead|scientist|architect|administrator|coordinator|specialist|"
    r"intern|officer|president|associate|accountant|nurse|teacher|professor|"
    r"technician|programmer|strategist|recruiter|representative|assistant|"
    r"executive|founder|owner|head|chief|vp|cto|ceo|cfo)\b",
    re.IGNORECASE,
)
# Split a header line into candidate parts. Only splits on strong separators
# (comma, pipe, en/em dash with spaces, hyphen with spaces, " at ") so hyphenated
# words like "Full-Stack" stay intact.
_HEADER_SPLIT_RE = re.compile(r"\s*[,|]\s*|\s+[–—]\s+|\s+-\s+|\s+at\s+", re.IGNORECASE)

# --- Section header patterns (curated, extensible) ----------------------------
# Maps a canonical section name to the header phrases that introduce it.
SECTION_HEADER_PATTERNS: dict[str, tuple[str, ...]] = {
    "summary": ("summary", "professional summary", "profile", "objective", "about"),
    "skills": (
        "skills",
        "technical skills",
        "core competencies",
        "technologies",
        "technical proficiencies",
    ),
    "experience": (
        "work experience",
        "professional experience",
        "experience",
        "employment history",
        "work history",
        "employment",
    ),
    "education": (
        "education",
        "academic background",
        "academic qualifications",
        "education and training",
    ),
    "contact": ("contact", "contact information", "contact details"),
}

_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?:\+?\d[\s\-.]?){7,}\d")

# A single date token: "Jan 2020" / "01/2020" / "2020".
_DATE_TOKEN = (
    r"(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}"
    r"|\d{1,2}[/\-]\d{4}|\d{4})"
)
_DATE_RANGE_RE = re.compile(
    rf"({_DATE_TOKEN})\s*(?:-|–|—|to)\s*({_DATE_TOKEN}|present|current|now)",
    re.IGNORECASE,
)
_YEARS_REQ_RE = re.compile(r"(\d+)\s*\+?\s*(?:years|yrs)", re.IGNORECASE)


def _load_seed_skills() -> list[str]:
    """Load the interim seed skill vocabulary (one skill per line, '#' comments)."""
    skills: list[str] = []
    for line in _SEED_SKILLS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            skills.append(line)
    return skills


# Build the PhraseMatcher once. attr="LOWER" gives case-insensitive matching; a
# match_id -> canonical-cased skill map restores canonical casing on output.
_SEED_SKILLS = _load_seed_skills()
_SKILL_MATCHER = PhraseMatcher(_NLP.vocab, attr="LOWER")
_CANONICAL_BY_KEY: dict[str, str] = {}
for _skill in _SEED_SKILLS:
    _key = f"SKILL::{_skill.lower()}"
    _SKILL_MATCHER.add(_key, [_NLP.make_doc(_skill)])
    _CANONICAL_BY_KEY[_key] = _skill


def _iso_partial(year: int, month: int | None) -> str:
    return f"{year:04d}-{month:02d}" if month else f"{year:04d}"


def _parse_date_token(token: str) -> tuple[int, int | None] | None:
    """Parse a single date token into (year, month|None). None if unparseable."""
    tok = token.strip().lower()
    m = re.match(r"(\d{1,2})[/\-](\d{4})", tok)
    if m:
        return int(m.group(2)), int(m.group(1))
    m = re.match(r"([a-z]{3,})\.?\s+(\d{4})", tok)
    if m:
        month = _MONTHS.get(m.group(1)[:3])
        return int(m.group(2)), month
    m = re.match(r"(\d{4})", tok)
    if m:
        return int(m.group(1)), None
    return None


def _to_date(year: int, month: int | None) -> date:
    return date(year, month or 1, 1)


class SectionSegmenter:
    """Splits resume raw text into named sections via headers, with a fallback."""

    def __init__(self) -> None:
        self.used_fallback: bool = False

    def segment(self, raw_text: str) -> dict[str, str]:
        """Return {section_name: text}. Unmatched text goes to 'unclassified'.

        Never drops text — only fails to categorize it. Sets ``used_fallback`` when
        no headers are found so the orchestrator can flag SECTION_HEADERS_NOT_DETECTED.
        """
        self.used_fallback = False
        lines = raw_text.splitlines()
        sections: dict[str, list[str]] = {}
        current = "unclassified"
        found_any_header = False

        for line in lines:
            header = self._match_header(line)
            if header is not None:
                current = header
                found_any_header = True
                sections.setdefault(current, [])
                continue
            sections.setdefault(current, []).append(line)

        if not found_any_header:
            # Fallback: no headers at all. Keep everything under 'unclassified';
            # the orchestrator routes the whole doc through each extractor as a
            # best effort. Flag it so downstream confidence reflects the guesswork.
            self.used_fallback = True

        return {name: "\n".join(body).strip() for name, body in sections.items()}

    def _match_header(self, line: str) -> str | None:
        """Return the canonical section a line introduces, if it's a header line."""
        stripped = line.strip().strip(":").lower()
        # Headers are short lines; a long sentence containing "experience" isn't one.
        if not stripped or len(stripped) > 40:
            return None
        for canonical, phrases in SECTION_HEADER_PATTERNS.items():
            if stripped in phrases:
                return canonical
        return None


class SkillExtractor:
    """Extracts a deterministic, deduplicated, canonically-cased skill list."""

    def __init__(self) -> None:
        self.warnings: list[ParsingWarningCode] = []

    def extract_skills(self, section_text: str) -> list[str]:
        self.warnings = []
        if not section_text.strip():
            return []
        doc = _NLP.make_doc(section_text)
        found: set[str] = set()
        for match_id, _start, _end in _SKILL_MATCHER(doc):
            key = _NLP.vocab.strings[match_id]
            found.add(_CANONICAL_BY_KEY[key])
        if not found:
            self.warnings.append(
                ParsingWarningCode.SKILL_SECTION_EMPTY_AFTER_EXTRACTION
            )
        # Sorted for determinism (PRD §8.3): same input → same ordered output.
        return sorted(found)


class ExperienceExtractor:
    """Extracts experience entries from an experience-section text block."""

    def __init__(self) -> None:
        self.warnings: list[ParsingWarningCode] = []

    def extract_experience(
        self, section_text: str, as_of: date | None = None
    ) -> list[ExperienceEntry]:
        """Extract experience entries. ``as_of`` pins "Present" to a fixed date for
        reproducibility (defaults to today)."""
        self.warnings = []
        reference = as_of or date.today()
        entries: list[ExperienceEntry] = []
        if not section_text.strip():
            return entries

        for block in self._split_blocks(section_text):
            range_match = _DATE_RANGE_RE.search(block)
            if range_match is None:
                # A block that looks like a job entry but has no parseable dates.
                if self._looks_like_job_block(block):
                    self.warnings.append(ParsingWarningCode.EXPERIENCE_DATES_AMBIGUOUS)
                continue
            entry = self._build_entry(block, range_match, reference)
            if entry is not None:
                entries.append(entry)
        return entries

    def _split_blocks(self, text: str) -> list[str]:
        blocks = re.split(r"\n\s*\n", text.strip())
        return [b.strip() for b in blocks if b.strip()]

    def _looks_like_job_block(self, block: str) -> bool:
        doc = _NLP(block)
        return any(ent.label_ == "ORG" for ent in doc.ents)

    def _build_entry(
        self, block: str, range_match: re.Match[str], as_of: date
    ) -> ExperienceEntry | None:
        start_parsed = _parse_date_token(range_match.group(1))
        if start_parsed is None:
            self.warnings.append(ParsingWarningCode.EXPERIENCE_DATES_AMBIGUOUS)
            return None
        start_year, start_month = start_parsed
        start_iso = _iso_partial(start_year, start_month)
        start_dt = _to_date(start_year, start_month)

        end_raw = range_match.group(2).lower()
        if end_raw in ("present", "current", "now"):
            end_iso: str | None = None
            end_dt = as_of
        else:
            end_parsed = _parse_date_token(range_match.group(2))
            if end_parsed is None:
                end_iso, end_dt = None, as_of
            else:
                end_iso = _iso_partial(*end_parsed)
                end_dt = _to_date(*end_parsed)

        years = round(max((end_dt - start_dt).days, 0) / 365.25, 2)

        title, company = self._extract_title_company(block, range_match)
        return ExperienceEntry(
            title=title,
            company=company,
            start_date=start_iso,
            end_date=end_iso,
            description=block,
            years_calculated=years,
        )

    def _extract_title_company(
        self, block: str, range_match: re.Match[str]
    ) -> tuple[str | None, str | None]:
        """Deterministic, order-independent title/company split.

        Classifies each header part by strong signals (company suffix like "Inc"
        / "Corporation"; role keyword like "Engineer" / "Manager") rather than
        trusting spaCy's ORG tag, which frequently mislabels a job title as an
        organization. spaCy ORG is used only as a last-resort company fallback.
        Leaves a field None when not confident (honesty over guessing).
        """
        header_line = next((ln for ln in block.splitlines() if ln.strip()), "")
        # Remove any inline date range so it can't pollute the parts.
        header_no_date = _DATE_RANGE_RE.sub("", header_line).strip()
        parts = [
            p.strip()
            for p in _HEADER_SPLIT_RE.split(header_no_date)
            if p.strip() and re.search(r"[A-Za-z]{2,}", p)
        ]
        if not parts:
            return None, None

        company = next((p for p in parts if _COMPANY_SUFFIX_RE.search(p)), None)
        title = next(
            (p for p in parts if p != company and _TITLE_KEYWORD_RE.search(p)), None
        )

        # If a title keyword was found but no suffix-company, the remaining part
        # (if any) is the company candidate.
        if company is None and title is not None:
            leftovers = [p for p in parts if p != title]
            company = leftovers[0] if leftovers else None
        # If a company was found but no title keyword, the remaining part is the
        # title candidate.
        if title is None and company is not None:
            leftovers = [p for p in parts if p != company]
            title = leftovers[0] if leftovers else None
        # Last resort for company: spaCy ORG, but never equal to the chosen title.
        if company is None:
            doc = _NLP(header_no_date)
            orgs = [
                ent.text.strip()
                for ent in doc.ents
                if ent.label_ == "ORG" and ent.text.strip() != title
            ]
            company = orgs[0] if orgs else None

        return title, company


class EducationExtractor:
    """Extracts education entries: degree level, institution, graduation year."""

    def __init__(self) -> None:
        self.warnings: list[ParsingWarningCode] = []

    def extract_education(
        self, section_text: str, as_of: date | None = None
    ) -> list[EducationEntry]:
        self.warnings = []
        entries: list[EducationEntry] = []
        if not section_text.strip():
            return entries

        current_year = (as_of or date.today()).year
        # Split on BLANK lines only so a degree, its institution, and its year
        # (typically on adjacent lines) stay in one block instead of fragmenting.
        for block in re.split(r"\n\s*\n", section_text.strip()):
            block = block.strip()
            if not block:
                continue
            degree = self._detect_degree(block)
            # Require a degree keyword to create an entry. Institution-only or
            # year-only blocks are almost always noise from a mistagged ORG (a
            # skill/company) in the header-less fallback path — dropping them
            # kills the false positives found in Phase 1.4 testing.
            if degree is None:
                continue
            year = self._detect_grad_year(block, current_year)
            institution = self._detect_institution(block)
            entries.append(
                EducationEntry(
                    degree=degree,
                    institution=institution,
                    field_of_study=None,
                    graduation_year=year,
                )
            )
        return entries

    def _detect_degree(self, text: str) -> str | None:
        low = text.lower()
        for canonical, keywords in DEGREE_KEYWORDS:
            if any(kw in low for kw in keywords):
                return canonical
        return None

    def _detect_grad_year(self, text: str, current_year: int) -> int | None:
        for match in re.findall(r"\b(19[5-9]\d|20\d\d)\b", text):
            year = int(match)
            if 1950 <= year <= current_year + 1:
                return year
        return None

    def _detect_institution(self, text: str) -> str | None:
        doc = _NLP(text)
        orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]
        return orgs[0] if orgs else None


class TotalExperienceCalculator:
    """Sums experience years across entries WITHOUT double-counting overlaps."""

    def calculate_total_years(
        self, experience_entries: list[ExperienceEntry], as_of: date | None = None
    ) -> float | None:
        """Merge overlapping date intervals before summing.

        Naive summation (adding each entry's years_calculated) would double-count
        concurrent roles — e.g. two overlapping 2018-2020 jobs would report 4
        years of experience for a 2-year span. We merge intervals first, then sum
        the merged spans, so overlapping months are counted once.

        ``as_of`` pins open-ended ("Present") roles to a fixed date for
        reproducibility (defaults to today).
        """
        reference = as_of or date.today()
        intervals: list[tuple[date, date]] = []
        for entry in experience_entries:
            start = self._parse_iso(entry.start_date)
            if start is None:
                continue  # Can't place an interval without a start.
            end = self._parse_iso(entry.end_date) or reference
            if end < start:
                continue
            intervals.append((start, end))

        if not intervals:
            return None

        intervals.sort(key=lambda iv: iv[0])
        merged: list[tuple[date, date]] = [intervals[0]]
        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:  # Overlaps (or touches) the current merged span.
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        total_days = sum((end - start).days for start, end in merged)
        return round(total_days / 365.25, 2)

    def _parse_iso(self, iso: str | None) -> date | None:
        if not iso:
            return None
        m = re.match(r"(\d{4})(?:-(\d{2}))?$", iso)
        if not m:
            return None
        year = int(m.group(1))
        month = int(m.group(2)) if m.group(2) else 1
        return date(year, month, 1)


class JobDescriptionStructurer:
    """Structures a JD into required vs preferred skills + requirements."""

    def __init__(self) -> None:
        self._skills = SkillExtractor()

    def structure(self, raw_text: str) -> dict[str, object]:
        required_lines: list[str] = []
        preferred_lines: list[str] = []
        neutral_lines: list[str] = []

        for line in raw_text.splitlines():
            low = line.lower()
            if any(sig in low for sig in ("preferred", "nice to have", "bonus")):
                preferred_lines.append(line)
            elif any(sig in low for sig in ("required", "must have", "requirement")):
                required_lines.append(line)
            else:
                neutral_lines.append(line)

        preferred = set(self._skills.extract_skills("\n".join(preferred_lines)))
        # Conservative default: skills without an explicit signal → required.
        required = set(
            self._skills.extract_skills("\n".join(required_lines + neutral_lines))
        )
        preferred -= required  # A skill is not both.

        return {
            "required_skills": sorted(required),
            "preferred_skills": sorted(preferred),
            "required_years_experience": self._extract_required_years(raw_text),
            "required_education_level": self._extract_required_education(raw_text),
        }

    def _extract_required_years(self, text: str) -> float | None:
        match = _YEARS_REQ_RE.search(text)
        return float(match.group(1)) if match else None

    def _extract_required_education(self, text: str) -> str | None:
        low = text.lower()
        for canonical, keywords in DEGREE_KEYWORDS:
            if any(kw in low for kw in keywords):
                return canonical
        return None


def _detect_contact_info_present(text: str) -> bool:
    """Return whether an email or phone appears in the text.

    PRIVACY: the matched PII string is intentionally never stored, logged, or
    returned — only presence is recorded, per PRD §9. The match objects go out of
    scope the moment this function returns its boolean.
    """
    return bool(_EMAIL_RE.search(text) or _PHONE_RE.search(text))


def _section_or_full(sections: dict[str, str], name: str, full_text: str) -> str:
    """Return a section's text, falling back to the full doc when it's absent.

    Used when header segmentation failed — extractors then scan the whole
    document rather than receiving nothing.
    """
    value = sections.get(name)
    return value if value else full_text


def structure_resume(
    extraction_result: ExtractionResult, as_of: date | None = None
) -> ParsedResume:
    """Top-level entry point: raw extraction → populated ParsedResume.

    ``as_of`` pins date math ("Present" roles, year bounds) to a fixed reference
    for reproducible evaluation (Phase 5); defaults to today for normal use.
    parsing_confidence is computed here via the Phase 1.3 calculators.
    """
    reference = as_of or date.today()
    raw_text = extraction_result.raw_text
    warnings: list[ParsingWarningCode] = list(extraction_result.warnings)

    segmenter = SectionSegmenter()
    sections = segmenter.segment(raw_text)
    if segmenter.used_fallback:
        warnings.append(ParsingWarningCode.SECTION_HEADERS_NOT_DETECTED)

    skill_ex = SkillExtractor()
    skills = skill_ex.extract_skills(_section_or_full(sections, "skills", raw_text))
    warnings.extend(skill_ex.warnings)

    exp_text = sections.get("experience")
    if not exp_text:
        warnings.append(ParsingWarningCode.NO_EXPERIENCE_SECTION_FOUND)
        exp_text = raw_text
    exp_ex = ExperienceExtractor()
    experience = exp_ex.extract_experience(exp_text, as_of=reference)
    warnings.extend(exp_ex.warnings)

    edu_ex = EducationExtractor()
    education = edu_ex.extract_education(
        _section_or_full(sections, "education", raw_text), as_of=reference
    )
    warnings.extend(edu_ex.warnings)

    total_years = TotalExperienceCalculator().calculate_total_years(
        experience, as_of=reference
    )
    contact_present = _detect_contact_info_present(raw_text)

    resume = ParsedResume(
        raw_text=raw_text,
        skills=skills,
        experience=experience,
        education=education,
        total_years_experience=total_years,
        contact_info_present=contact_present,
        parsing_confidence=0.0,  # Overwritten below with the real 1.3 score.
        parsing_warnings=[w.value for w in dict.fromkeys(warnings)],
        pipeline_version=PARSER_PIPELINE_VERSION,
    )
    # Phase 1.3: compute the real confidence from completeness of the object above.
    confidence = calculate_resume_parsing_confidence(extraction_result, resume)
    return resume.model_copy(update={"parsing_confidence": confidence})


def structure_job_description(
    extraction_result: ExtractionResult,
) -> ParsedJobDescription:
    """Top-level entry point: raw extraction → populated ParsedJobDescription."""
    raw_text = extraction_result.raw_text
    structured = JobDescriptionStructurer().structure(raw_text)
    jd = ParsedJobDescription(
        raw_text=raw_text,
        required_skills=structured["required_skills"],  # type: ignore[arg-type]
        preferred_skills=structured["preferred_skills"],  # type: ignore[arg-type]
        required_years_experience=structured[  # type: ignore[arg-type]
            "required_years_experience"
        ],
        required_education_level=structured[  # type: ignore[arg-type]
            "required_education_level"
        ],
        parsing_confidence=0.0,  # Overwritten below with the real 1.3 score.
        pipeline_version=PARSER_PIPELINE_VERSION,
    )
    confidence = calculate_jd_parsing_confidence(extraction_result, jd)
    return jd.model_copy(update={"parsing_confidence": confidence})

"""Parser validation report generator (Phase 1.4 deliverable).

Runs every e2e fixture through the full pipeline and prints a clean summary
table suitable for pasting into the capstone report's Testing & Validation
section (PRD §14 item 6).

Usage:
    python -m tests.integration.generate_parser_report
"""

from __future__ import annotations

from app.services.confidence.confidence_utils import confidence_to_band
from tests.integration.test_parser_pipeline_e2e import (
    CAREER_SWITCHER_RESUME,
    CLEAN_RESUME,
    COLUMN_BLEED_RESUME,
    INLINE_SKILLS_RESUME,
    NON_NATIVE_RESUME,
    SPARSE_RESUME,
    TYPO_RESUME,
    _run_pipeline,
)

_FIXTURES = [
    ("1 clean", CLEAN_RESUME),
    ("2 column-bleed", COLUMN_BLEED_RESUME),
    ("3 typos", TYPO_RESUME),
    ("4 non-native", NON_NATIVE_RESUME),
    ("5 career-switcher", CAREER_SWITCHER_RESUME),
    ("6 sparse", SPARSE_RESUME),
    ("7 inline-skills", INLINE_SKILLS_RESUME),
]


def generate_report() -> str:
    header = (
        f"{'fixture':<18} {'conf':>5} {'band':>7} "
        f"{'skills':>6} {'exp':>4} {'edu':>4}  warnings"
    )
    lines = [header, "-" * len(header)]
    for name, text in _FIXTURES:
        r = _run_pipeline(text)
        band = confidence_to_band(r.parsing_confidence).value
        warns = ", ".join(r.parsing_warnings) or "-"
        lines.append(
            f"{name:<18} {r.parsing_confidence:>5.2f} {band:>7} "
            f"{len(r.skills):>6} {len(r.experience):>4} {len(r.education):>4}  {warns}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(generate_report())

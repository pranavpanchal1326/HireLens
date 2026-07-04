"""Blind multi-rater tooling (Phase 5.1).

Generates one BLIND CSV per rater from a frozen curated pair list, and ingests the
filled-in sheets back into GroundTruthPair objects. Blind + independent is
mandatory (PRD §6): each rater fills their own sheet without seeing others'.

Minimal by design — a spreadsheet-style CSV workflow, NOT a UI product.
"""

from __future__ import annotations

import csv
from pathlib import Path

from app.services.evaluation.ground_truth_schema import (
    CaseType,
    GroundTruthPair,
    RaterScore,
)

_SHEET_COLUMNS = [
    "pair_id",
    "resume_id",
    "jd_id",
    "case_type",
    "score",
    "justification",
]


class CuratedPair:
    """A frozen, human-curated pair (before any rating). Not a score-bearing type."""

    def __init__(
        self, pair_id: str, resume_id: str, jd_id: str, case_type: CaseType
    ) -> None:
        self.pair_id = pair_id
        self.resume_id = resume_id
        self.jd_id = jd_id
        self.case_type = case_type


def write_blind_rating_sheets(
    curated_pairs: list[CuratedPair], rater_ids: list[str], out_dir: str
) -> list[str]:
    """Write one blank rating CSV per rater (score/justification empty). Returns the
    written file paths. Raters fill these INDEPENDENTLY and BLIND."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for rater_id in rater_ids:
        path = out / f"ratings_{rater_id}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_SHEET_COLUMNS)
            writer.writeheader()
            for cp in curated_pairs:
                writer.writerow(
                    {
                        "pair_id": cp.pair_id,
                        "resume_id": cp.resume_id,
                        "jd_id": cp.jd_id,
                        "case_type": cp.case_type,
                        "score": "",  # rater fills
                        "justification": "",  # rater fills
                    }
                )
        written.append(str(path))
    return written


def load_rater_sheet(path: str, rater_id: str) -> dict[str, RaterScore]:
    """Load one filled rater sheet → {pair_id: RaterScore}. Rows with an empty
    score are skipped (that rater hasn't scored that pair yet)."""
    result: dict[str, RaterScore] = {}
    with Path(path).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw_score = (row.get("score") or "").strip()
            if not raw_score:
                continue
            result[row["pair_id"]] = RaterScore(
                rater_id=rater_id,
                score=float(raw_score),
                justification=(row.get("justification") or "").strip(),
            )
    return result


def assemble_pairs(
    curated_pairs: list[CuratedPair], rater_sheets: dict[str, dict[str, RaterScore]]
) -> list[GroundTruthPair]:
    """Combine curated pairs + per-rater loaded sheets into GroundTruthPair objects
    (unreconciled). ``rater_sheets`` maps rater_id -> {pair_id: RaterScore}."""
    pairs: list[GroundTruthPair] = []
    for cp in curated_pairs:
        scores = [
            sheet[cp.pair_id] for sheet in rater_sheets.values() if cp.pair_id in sheet
        ]
        pairs.append(
            GroundTruthPair(
                pair_id=cp.pair_id,
                resume_id=cp.resume_id,
                jd_id=cp.jd_id,
                case_type=cp.case_type,
                rater_scores=scores,
                # Always awaiting until reconcile_dataset runs — never pre-mark
                # reconciled without the reconciliation step actually running.
                status="awaiting_raters",
            )
        )
    return pairs

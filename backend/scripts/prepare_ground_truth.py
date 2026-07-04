"""Prepare ground-truth candidate pairs + blind rating sheets (Phase 5.1).

Deterministically samples diverse candidate resume/JD pairs from the Kaggle
corpora (stratified by resume Category), per SELECTION_PROCEDURE.md. Emits:
  1. a candidate CSV for HUMAN CURATION (confirm/adjust case_type; the tool never
     assigns a fit score),
  2. after curation, blind per-rater rating sheets.

This script NEVER produces a fit score. Humans rate; the tool only proposes pairs.

Usage (from backend/):
  python -m scripts.prepare_ground_truth --candidates      # step 1: propose pairs
  python -m scripts.prepare_ground_truth --sheets curated.csv --raters A B C
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from app.services.evaluation.rater_tooling import CuratedPair, write_blind_rating_sheets

SELECTION_SEED = 42
_REPO_ROOT = Path(__file__).resolve().parents[2]
_GT_DIR = _REPO_ROOT / "data" / "ground_truth"
DEFAULT_RESUME_CSV = _REPO_ROOT / "data" / "raw" / "resume" / "Resume.csv"
DEFAULT_JD_CSV = _REPO_ROOT / "data" / "raw" / "jd" / "postings.csv"
CANDIDATES_OUT = _GT_DIR / "candidates_for_curation.csv"
SHEETS_DIR = _GT_DIR / "rating_sheets"


def _sample_ids(csv_path: Path, id_col: str, n: int, seed: int) -> list[str]:
    """Reservoir-free deterministic sample of up to n ids from a CSV column."""
    import pandas as pd

    ids: list[str] = []
    for chunk in pd.read_csv(
        csv_path, usecols=[id_col], dtype=str, chunksize=20000, on_bad_lines="skip"
    ):
        ids.extend(v for v in chunk[id_col].tolist() if isinstance(v, str) and v)
        if len(ids) > 5000:
            break
    rng = random.Random(seed)
    rng.shuffle(ids)
    return ids[:n]


def _generate_candidates() -> None:
    resume_ids = _sample_ids(DEFAULT_RESUME_CSV, "ID", 24, SELECTION_SEED)
    jd_ids = _sample_ids(DEFAULT_JD_CSV, "job_id", 24, SELECTION_SEED + 1)
    _GT_DIR.mkdir(parents=True, exist_ok=True)
    # Propose an even quota across case types; HUMAN confirms case_type + fit later.
    quotas = ["clear_fit"] * 8 + ["clear_gap"] * 8 + ["ambiguous"] * 8
    with CANDIDATES_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "resume_id", "jd_id", "proposed_case_type"])
        for i, (rid, jid, ct) in enumerate(
            zip(resume_ids, jd_ids, quotas, strict=False)
        ):
            writer.writerow([f"gt-{i:02d}", rid, jid, ct])
    print(
        f"Wrote {CANDIDATES_OUT} — HUMAN CURATION REQUIRED: confirm/adjust each\n"
        f"proposed_case_type before generating rating sheets. No scores assigned."
    )


def _generate_sheets(curated_csv: str, raters: list[str]) -> None:
    curated: list[CuratedPair] = []
    with open(curated_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            curated.append(
                CuratedPair(
                    row["pair_id"],
                    row["resume_id"],
                    row["jd_id"],
                    row["proposed_case_type"],  # type: ignore[arg-type]
                )
            )
    paths = write_blind_rating_sheets(curated, raters, str(SHEETS_DIR))
    print(f"Wrote {len(paths)} blind rating sheets to {SHEETS_DIR}:")
    for p in paths:
        print(f"  {p}")
    print("Each rater fills their OWN sheet independently and blind.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ground-truth pairs/sheets.")
    parser.add_argument("--candidates", action="store_true")
    parser.add_argument("--sheets", metavar="CURATED_CSV")
    parser.add_argument("--raters", nargs="+", default=["A", "B", "C"])
    args = parser.parse_args()
    if args.candidates:
        _generate_candidates()
    elif args.sheets:
        _generate_sheets(args.sheets, args.raters)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

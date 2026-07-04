"""Ingest filled rater sheets → reconciled ground-truth dataset (Phase 5.1).

Reads each rater's completed blind sheet, assembles GroundTruthPair objects, runs
multi-rater reconciliation (naive mean + inter-rater Pearson + divergence flags),
and persists the versioned dataset Phase 5.2 will consume.

Refuses to fabricate: if no rater sheets contain scores, the output stays in the
AWAITING_RATERS state rather than inventing numbers.

Usage (from backend/):
  python -m scripts.reconcile_ground_truth --curated curated.csv \
      --sheets data/ground_truth/rating_sheets/ratings_A.csv:A ...
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from app.services.evaluation.ground_truth_schema import (
    GroundTruthDataset,
    save_dataset,
)
from app.services.evaluation.rater_tooling import (
    CuratedPair,
    assemble_pairs,
    load_rater_sheet,
)
from app.services.evaluation.reconciliation import reconcile_dataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = _REPO_ROOT / "data" / "ground_truth" / "ground_truth_dataset.json"


def _load_curated(path: str) -> list[CuratedPair]:
    curated: list[CuratedPair] = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            curated.append(
                CuratedPair(
                    row["pair_id"],
                    row["resume_id"],
                    row["jd_id"],
                    row["proposed_case_type"],  # type: ignore[arg-type]
                )
            )
    return curated


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile rater sheets → dataset.")
    parser.add_argument("--curated", required=True)
    parser.add_argument(
        "--sheets", nargs="+", required=True, help="path:rater_id per sheet"
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    curated = _load_curated(args.curated)
    rater_sheets = {}
    for spec in args.sheets:
        path, rater_id = spec.rsplit(":", 1)
        rater_sheets[rater_id] = load_rater_sheet(path, rater_id)

    total_scores = sum(len(s) for s in rater_sheets.values())
    pairs = assemble_pairs(curated, rater_sheets)
    dataset = GroundTruthDataset(
        pairs=pairs,
        notes=f"seed=42; {len(rater_sheets)} rater sheet(s) ingested.",
    )
    dataset = reconcile_dataset(dataset)
    save_dataset(dataset, args.out)

    if total_scores == 0:
        print("AWAITING REAL RATER INPUT — no scores found; dataset saved in the")
        print("awaiting_raters state. No numbers were fabricated.")
    else:
        flagged = sum(1 for p in dataset.pairs if p.divergence_flag)
        print(
            f"Reconciled {len(dataset.pairs)} pairs | raters={dataset.n_raters} | "
            f"inter-rater agreement={dataset.overall_inter_rater_agreement} | "
            f"{flagged} divergence-flagged"
        )
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()

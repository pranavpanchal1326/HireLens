"""Ingest the raw ESCO skills CSV into a clean taxonomy (Phase 3.1).

Run ONCE after the ESCO bundle is in data/raw/esco/. Persists the cleaned
taxonomy to data/processed/ and prints preprocessing statistics — a citable
artifact for the capstone Data Collection & Preprocessing section (PRD §14 item 2).

Usage (from backend/):
  python -m scripts.ingest_esco_taxonomy
  python -m scripts.ingest_esco_taxonomy --csv <path> --out <path>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.services.rag.taxonomy_ingestion import ESCOTaxonomyIngester, save_taxonomy

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = _REPO_ROOT / "data" / "raw" / "esco" / "skills_en.csv"
DEFAULT_OUT = _REPO_ROOT / "data" / "processed" / "esco_taxonomy.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the ESCO skills taxonomy.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print(f"Ingesting ESCO taxonomy from {args.csv}")
    ingester = ESCOTaxonomyIngester(args.csv)
    entries = ingester.ingest()
    save_taxonomy(entries, args.out)

    total_alt = sum(len(e.alt_labels) for e in entries)
    avg_alt = total_alt / len(entries) if entries else 0.0
    print(
        f"Concepts ingested:   {len(entries)}\n"
        f"Total alt-labels:    {total_alt}\n"
        f"Avg alt-labels/skill:{avg_alt:>7.2f}\n"
        f"Rows skipped:        {ingester.last_skipped_count}\n"
        f"Saved -> {args.out}"
    )


if __name__ == "__main__":
    main()

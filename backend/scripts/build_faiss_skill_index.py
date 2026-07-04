"""Build and persist the FAISS skill index from the ESCO taxonomy (Phase 3.2).

Run ONCE after scripts.ingest_esco_taxonomy. Loads the cleaned taxonomy, embeds
every label + synonym, builds a normalized FAISS index, and persists index +
metadata to data/processed/. Prints stats — a citable artifact for the report.

Usage (from backend/):
  python -m scripts.build_faiss_skill_index
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from app.services.rag.faiss_index_builder import FAISSSkillIndexBuilder, save_index
from app.services.rag.taxonomy_ingestion import load_taxonomy
from app.services.scoring.embedding_scorer import EmbeddingScorer

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED = _REPO_ROOT / "data" / "processed"
DEFAULT_TAXONOMY = _PROCESSED / "esco_taxonomy.jsonl"
DEFAULT_INDEX = _PROCESSED / "esco_skill.index"
DEFAULT_METADATA = _PROCESSED / "esco_skill_metadata.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FAISS skill index.")
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    args = parser.parse_args()

    print(f"Loading taxonomy from {args.taxonomy}")
    entries = load_taxonomy(args.taxonomy)
    print(f"Loaded {len(entries)} concepts. Embedding labels + synonyms...")

    builder = FAISSSkillIndexBuilder(EmbeddingScorer())
    start = time.time()
    index, metadata = builder.build_index(entries)
    elapsed = time.time() - start

    save_index(index, metadata, args.index, args.metadata)

    preferred = sum(1 for m in metadata if m.label_type == "preferred")
    alt = sum(1 for m in metadata if m.label_type == "alt")
    size_mb = Path(args.index).stat().st_size / (1024 * 1024)
    print(
        f"Total vectors indexed: {index.ntotal}\n"
        f"  preferred-label vectors: {preferred}\n"
        f"  alt-label vectors:       {alt}\n"
        f"Index file size: {size_mb:.2f} MB\n"
        f"Build time:      {elapsed:.1f}s\n"
        f"Saved -> {args.index} / {args.metadata}"
    )


if __name__ == "__main__":
    main()

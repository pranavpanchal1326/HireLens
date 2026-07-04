"""HireLens dataset downloader.

Downloads the three externally-sourced datasets defined in PRD §6 into
``data/raw/``. The fourth dataset (ground truth) is self-built in Phase 5 and is
NOT handled here.

Datasets (PRD §6 — deliberately complete, do not add more):
  1. Resume corpus (Kaggle)          -> data/raw/resume/
  2. JD corpus (Kaggle LinkedIn)     -> data/raw/jd/
  3. Skill taxonomy (ESCO)           -> data/raw/esco/

Usage:
  python scripts/download_data.py --all
  python scripts/download_data.py --resume --jd      # Kaggle only
  python scripts/download_data.py --esco             # ESCO only

Prerequisites:
  Kaggle datasets require an API token. Create one at
  https://www.kaggle.com/settings/account -> "Create New Token", then place the
  downloaded kaggle.json at:
      Windows:  %USERPROFILE%\\.kaggle\\kaggle.json
      Unix:     ~/.kaggle/kaggle.json
  (or set KAGGLE_USERNAME / KAGGLE_KEY environment variables).

  ESCO: the official CSV bundle now sits behind an EU Login on the portal. If you
  have downloaded it manually, drop the .zip into data/raw/esco/ and re-run with
  --esco to extract it. Otherwise this script falls back to a note explaining the
  manual step. No paid access is ever required.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

# --- Configuration -----------------------------------------------------------
# Kaggle dataset slugs. Confirm these match the exact datasets you intend to use;
# change here if your PRD points at different specific Kaggle datasets.
RESUME_KAGGLE_SLUG = "snehaanbhawal/resume-dataset"
JD_KAGGLE_SLUG = "arshkon/linkedin-job-postings"

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"
RESUME_DIR = DATA_RAW / "resume"
JD_DIR = DATA_RAW / "jd"
ESCO_DIR = DATA_RAW / "esco"


def _download_kaggle(slug: str, dest: Path) -> None:
    """Download and unzip a Kaggle dataset into ``dest`` via the Kaggle CLI."""
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[kaggle] downloading {slug} -> {dest}")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "kaggle",
                "datasets",
                "download",
                "-d",
                slug,
                "-p",
                str(dest),
                "--unzip",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(
            f"[kaggle] FAILED for {slug}. Ensure kaggle.json credentials are in "
            f"place (see module docstring). Exit code {exc.returncode}.",
            file=sys.stderr,
        )
        raise


def download_resume() -> None:
    _download_kaggle(RESUME_KAGGLE_SLUG, RESUME_DIR)


def download_jd() -> None:
    _download_kaggle(JD_KAGGLE_SLUG, JD_DIR)


def download_esco() -> None:
    """Extract a manually-provided ESCO CSV bundle from data/raw/esco/.

    The ESCO portal gates its bulk CSV behind a (free) EU Login, so this script
    does not scrape it. Download the "classification - en - csv" bundle from
    https://esco.ec.europa.eu/en/use-esco/download, drop the .zip into
    data/raw/esco/, then re-run with --esco.
    """
    ESCO_DIR.mkdir(parents=True, exist_ok=True)
    zips = sorted(ESCO_DIR.glob("*.zip"))
    if not zips:
        print(
            "[esco] No .zip found in data/raw/esco/.\n"
            "       Download the ESCO 'classification - en - csv' bundle from\n"
            "       https://esco.ec.europa.eu/en/use-esco/download (free EU Login),\n"
            "       place the .zip in data/raw/esco/, and re-run with --esco."
        )
        return
    for archive in zips:
        print(f"[esco] extracting {archive.name}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(ESCO_DIR)
    print(f"[esco] done -> {ESCO_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HireLens datasets.")
    parser.add_argument("--all", action="store_true", help="Download everything.")
    parser.add_argument("--resume", action="store_true", help="Kaggle resume corpus.")
    parser.add_argument("--jd", action="store_true", help="Kaggle LinkedIn JD corpus.")
    parser.add_argument("--esco", action="store_true", help="ESCO skill taxonomy.")
    args = parser.parse_args()

    if not any([args.all, args.resume, args.jd, args.esco]):
        parser.print_help()
        return

    if args.all or args.resume:
        download_resume()
    if args.all or args.jd:
        download_jd()
    if args.all or args.esco:
        download_esco()


if __name__ == "__main__":
    main()

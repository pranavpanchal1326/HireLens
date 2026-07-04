# HireLens Data

Per **PRD §6**, HireLens uses exactly **four** datasets — no more (adding unused
datasets is scope-creep that §7.3 explicitly penalizes). Three are downloaded
here; the fourth is self-built in Phase 5.

| # | Dataset | Source | Purpose | Location | Status |
|---|---------|--------|---------|----------|--------|
| 1 | Resume corpus | Kaggle | Parser testing, TF-IDF fitting | `data/raw/resume/` | needs Kaggle token |
| 2 | JD corpus | Kaggle (LinkedIn job postings) | JD-side TF-IDF fitting | `data/raw/jd/` | needs Kaggle token |
| 3 | Skill taxonomy | ESCO (EU) | RAG skill-matcher vocabulary | `data/raw/esco/` | manual (EU Login) |
| 4 | Ground truth | Self-built | Model training + evaluation | `data/raw/ground_truth/` → labels | **Phase 5** |

Raw data contents are git-ignored (only folder structure via `.gitkeep` is tracked).

## How to download (1–3)

From the repo root, using the backend venv's Python:

```bash
# Kaggle datasets (requires kaggle.json — see below)
./backend/.venv/Scripts/python.exe scripts/download_data.py --resume --jd

# ESCO (after manually placing the CSV zip in data/raw/esco/)
./backend/.venv/Scripts/python.exe scripts/download_data.py --esco

# everything at once
./backend/.venv/Scripts/python.exe scripts/download_data.py --all
```

### Kaggle API token (datasets 1 & 2)

Kaggle refuses anonymous dataset downloads. Create a token at
<https://www.kaggle.com/settings/account> → **Create New Token**, then place
`kaggle.json` at:

- Windows: `%USERPROFILE%\.kaggle\kaggle.json`
- Unix: `~/.kaggle/kaggle.json`

Default slugs used by the script (edit in `scripts/download_data.py` if your PRD
points at different specific datasets):

- Resume: `snehaanbhawal/resume-dataset`
- JD: `arshkon/linkedin-job-postings`

### ESCO skill taxonomy (dataset 3)

The ESCO portal now gates its bulk CSV behind a free **EU Login**. Download the
`classification - en - csv` bundle from
<https://esco.ec.europa.eu/en/use-esco/download>, drop the `.zip` into
`data/raw/esco/`, and run the `--esco` step to extract it. No paid access needed.

### Ground truth (dataset 4)

Self-built in **Phase 5** using the `GroundTruthLabel` schema
(`backend/app/schemas/feedback.py`) with multi-rater reconciliation. Not
downloaded.

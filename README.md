# HireLens

HireLens is a two-sided (job seeker + recruiter) AI resume intelligence platform
that parses resumes and job descriptions, scores fit with a hybrid TF-IDF +
embedding model, and surfaces explainable skill matches. Built entirely on free,
open-source tooling — no paid services or API keys required to run the base system.

> This is Phase 0.1 (repo scaffold). Business logic arrives in later phases.

## Tech Stack

> Placeholder table — align with your PRD §11. Only Phase 0.1 dependencies are
> installed today; later rows are added when their phase is built.

| Layer            | Technology                                   | Phase |
| ---------------- | -------------------------------------------- | ----- |
| API              | FastAPI, Uvicorn                             | 0.1   |
| Config           | pydantic-settings, python-dotenv             | 0.1   |
| Parsing          | spaCy, pdfplumber / PyMuPDF                  | later |
| Scoring          | TF-IDF + embeddings (hybrid)                 | later |
| RAG matching     | FAISS / Chroma, sentence-transformers        | later |
| Orchestration    | Rule-based agent                             | later |
| ML re-ranking    | scikit-learn / XGBoost                        | later |
| Storage          | SQLite (dev) / Supabase (prod)               | later |
| Frontend         | React                                        | 8     |
| Tooling          | pytest, black, ruff, mypy                    | 0.1   |

## Local Setup

All commands run from the `backend/` directory.

```bash
cd backend

# 1. Create and activate a virtual environment (Python 3.11+)
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 2. Install dependencies (dev deps include runtime deps)
pip install -r requirements-dev.txt

# 3. Configure environment
cp .env.example .env    # then edit as needed

# 4. Run the API
uvicorn app.main:app --reload
# Health check: http://127.0.0.1:8000/health

# 5. Run the tests
pytest
```

## Folder Structure

```
hirelens/
├── backend/     # FastAPI service (app/, tests/, Dockerfile, requirements)
├── frontend/    # React app — scaffolded in Phase 8
├── data/        # raw / processed / ground_truth data lake (git-tracked, contents ignored)
├── docs/        # ARCHITECTURE.md and design references
├── README.md
└── CHANGELOG.md
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the system architecture.

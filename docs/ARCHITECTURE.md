# HireLens — System Architecture

> **PLACEHOLDER — replace with the verbatim diagram from your PRD Section 4.**
> The scaffold prompt referenced pasting the PRD's canonical architecture diagram
> here. I did not have the PRD file, so the diagram below is a faithful
> reconstruction from the described components. Swap it for the exact PRD text
> before treating this as canonical.

```
                            HireLens — High-Level Architecture

  Job Seeker ─┐                                          ┌─ Recruiter
              │                                          │
              ▼                                          ▼
        ┌───────────────────────────────────────────────────────┐
        │                    Frontend (React)                    │   [Phase 8]
        └───────────────────────────┬───────────────────────────┘
                                     │ REST (/api/v1)
                                     ▼
        ┌───────────────────────────────────────────────────────┐
        │                   FastAPI Backend                      │
        │                                                        │
        │   ┌──────────────┐   ┌──────────────┐  ┌────────────┐  │
        │   │   Parser     │──▶│   Scorer     │─▶│ RAG Skill  │  │
        │   │ spaCy +      │   │ TF-IDF +     │  │ Matcher    │  │
        │   │ pdfplumber / │   │ embeddings   │  │ FAISS /    │  │
        │   │ PyMuPDF      │   │ (hybrid)     │  │ Chroma     │  │
        │   └──────────────┘   └──────────────┘  └─────┬──────┘  │
        │                                              │         │
        │            ┌──────────────────────┐          │         │
        │            │  Agent Orchestrator   │◀─────────┘         │
        │            │  (rule-based)         │                    │
        │            └───────────┬───────────┘                    │
        │                        ▼                                │
        │            ┌──────────────────────┐                     │
        │            │  ML Re-ranker         │                     │
        │            │  scikit-learn/XGBoost │                     │
        │            └──────────────────────┘                     │
        └───────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────┐
                     │   Storage (SQLite / Supabase) │
                     └───────────────────────────┘

  Data lake:  data/raw  ──▶  data/processed  ──▶  data/ground_truth
```

All components are free / open-source. No paid API keys are required to run the
base system.

## Pipeline Versioning

HireLens tags every produced score/ranking with a named, reproducible **pipeline
version**, so any result can be traced back to the exact configuration of active
scoring components that produced it.

**Why this exists.** PRD §7.2 mandates an ablation study comparing exactly five
pipeline stages, and PRD §8.2 requires accuracy improvements to be demonstrable
version-over-version. Both depend on being able to answer "which components were
active for this score?" deterministically. The registry
(`backend/app/core/pipeline_registry.py`) is a static, version-controlled Python
object — deliberately not a mutable database table — so the evaluation methodology
is auditable and reproducible by a grader. Old configs are never deleted.

**The five LOCKED versions** (source of truth: `PipelineVersion` enum in
`backend/app/schemas/pipeline.py`):

| Version | Includes | Excludes |
|---|---|---|
| `v1-tfidf` | TF-IDF lexical similarity only | embeddings, RAG, ML |
| `v2-embeddings` | Embedding semantic similarity only | TF-IDF, RAG, ML |
| `v3-hybrid` | TF-IDF + embeddings | RAG, ML |
| `v4-hybrid-rag` | Hybrid + RAG skill matching | ML re-ranker |
| `v5-full-ml` | Full pipeline + trained ML re-ranker (production) | — |

**Enforcement rule.** `ScoreResult.pipeline_version` must always be one of these
five values. The `PipelineVersion` (str) enum enforces this at the schema level —
no component may write a version string that isn't an enum member, preventing
typo-fragmentation (e.g. `v1-tfidf` vs `v1_tfidf`) from silently splitting the
evaluation data. Exactly one version carries `is_active=True` at any time (the
current production default), enforced by `get_active_pipeline_version()`.

> Feature weights in the registry are honest **placeholders** until Phase 6 grid
> search populates tuned values. Each version zeroes every feature it does not
> activate, so `v1-tfidf`/`v2-embeddings` are never secretly hybrid.

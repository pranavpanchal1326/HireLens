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

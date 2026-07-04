# Ground-Truth Pair Selection Procedure (Phase 5.1)

A **reproducible, documented** procedure for choosing the 20–30 resume/JD pairs
that form the evaluation ground truth. Selection is deliberately DIVERSE, not
randomly convenient — a set of only easy pairs cannot validate the confidence-band
logic (Phase 4.3/4.4), and the PRD (§8.2) requires explicit edge-case coverage.

## Sources (fixed by PRD §6)
- Resumes: `data/raw/resume/Resume.csv` (Kaggle Resume Dataset; has a `Category`
  column — Engineering, HR, Finance, Designer, etc.).
- JDs: `data/raw/jd/postings.csv` (Kaggle LinkedIn Job Postings; `title` +
  `description`). No new scraping; no other dataset.

## Target composition (24 pairs, within the PRD's 20–30 range)
Stratified across three case types so results can later be sliced by difficulty:

| case_type   | count | what it stresses |
|-------------|------:|------------------|
| `clear_fit` | 8 | obviously strong matches (resume category ≈ JD domain, skills align) |
| `clear_gap` | 8 | obviously weak matches (unrelated category/domain) |
| `ambiguous` | 8 | transferable-but-nonexact skills, career-switchers, non-native phrasing, seniority-framing mismatch of similar substance |

## Deterministic procedure
1. **Fix a random seed** (`SELECTION_SEED = 42`) so the candidate draw is
   reproducible; record it in the dataset `notes`.
2. **Stratify resumes by `Category`** and JDs by a coarse domain bucket derived
   from `title`. `generate_candidate_pairs()` samples across buckets so no single
   domain dominates.
3. **Compose candidates by intended case_type:**
   - `clear_fit`: resume category matched to a same-domain JD.
   - `clear_gap`: resume category matched to a deliberately unrelated-domain JD.
   - `ambiguous`: cross-domain pairs with plausible transferable overlap (e.g. a
     hospitality-management resume against a project-coordination JD), plus any
     resumes flagged non-native/career-switch during Phase 1.4 testing.
4. **Human curation gate (mandatory).** The generated candidates are a STARTING
   POINT. A human reviewer confirms each pair actually belongs to its intended
   `case_type` before it enters the rating sheets — the tool proposes, a human
   disposes. The tool never assigns a fit score or a final case_type unilaterally.
5. **Freeze the curated pair list** (resume_id, jd_id, case_type) and generate the
   blind per-rater rating sheets from it.

## Why this is defensible in the report (PRD §14 §2)
- Reproducible (seeded) draw + explicit stratification targets.
- Edge cases are a *designed quota*, not an afterthought.
- The tool/human split keeps selection honest: automation ensures diversity;
  humans own every fit-relevant judgment.

# Name-Pair Selection Rationale (Phase 5.5 fairness harness)

The bias/fairness name-swap test (PRD §7.4) swaps identity-signaling names on a
resume, re-scores through the REAL pipeline, and reports whether the score moves
when it shouldn't. Finding bias here is a **valuable, reportable outcome** — not a
bug to quietly patch (PRD §7.4, Design Blueprint P3).

## Why a deliberate grid, not one pair
The PRD's illustrative "John → Priya" conflates gender AND ethnicity, so a delta
there can't be attributed to either alone. A defensible fairness methodology needs
pairs that **isolate** signals. The set below is drawn from the tradition of
resume-audit studies (e.g. Bertrand & Mullainathan 2004's name-signal design).

## Name pairs

| # | original → replacement | probes | isolation |
|---|------------------------|--------|-----------|
| 1 | John → Priya    | Western-male ↔ South-Asian-female | gender + ethnicity (PRD example) |
| 2 | Greg → Jamal    | Western-male ↔ African-American-male | ethnicity, **gender held** |
| 3 | Emily → Lakisha | Western-female ↔ African-American-female | ethnicity, **gender held** |
| 4 | Sarah → Fatima  | Western-female ↔ Arabic/Muslim-female | ethnicity/religion, **gender held** |
| 5 | Michael → Wei   | Western-male ↔ East-Asian-male | ethnicity, **gender held** |
| 6 | John → Juan     | Western-male ↔ Hispanic-male | ethnicity, **gender held** |
| 7 | John → Joan     | male ↔ female (both Western) | **gender isolated**, ethnicity held |

Pairs 2–7 hold one axis constant so a per-pair delta is interpretable; pair 1 is
kept because it is the PRD's own example and probes the combined signal.

## Interpretation rules (report, don't fix)
- Report the **full delta distribution and per-pair breakdown**, never a pooled
  mean alone (a near-zero mean can hide large opposite-direction deltas).
- Report how many trials actually **contained** the swapped name — a fairness
  claim over resumes that never carried the name is vacuous.
- If a bias is found, it goes in the report's fairness section (PRD §14 §6). It is
  NOT silently remediated by editing Phase 2/3 — that is separate, deliberate scope.

> Name pairs and this rationale are a human-reviewable methodology choice. No LLM
> selected them or judges the deltas — selection is documented here, significance
> is a statistical computation.

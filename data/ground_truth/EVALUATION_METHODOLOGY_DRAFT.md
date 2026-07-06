# Evaluation Methodology — Ground Truth & Inter-Rater Reliability

> **Status:** Draft for Report §14 (R13). Documentation only — contains no fit
> scores and prescribes none. The `{N}` and bracketed `{…}` markers are resolved
> once the actual number of completed rater sheets is known.

## 1. Overview

The evaluation ground truth is a curated set of **28 resume/JD pairs**, stratified
across three difficulty strata (`clear_fit`, `clear_gap`, `ambiguous`) per the
selection procedure. Each pair is scored **0–100 for fit** by one or more
independent human raters working **blind** against a shared anchored rubric.
Reconciliation produces a single ground-truth score per pair while preserving,
rather than averaging away, any material disagreement between raters.

No language model produces or influences any rating value; automation is confined
to pair generation, sheet assembly, and statistical reconciliation.

## 2. Rater Count and Its Statistical Implications

The target was three independent raters. The pipeline is deliberately
**rater-count-agnostic** — it reconciles and reports correctly for whatever number
of sheets are completed — so the methodology is stated below for each realized
outcome. In keeping with the project's *honest-over-impressive* principle
(Design Blueprint P3), the report states the **actual** rater count and its
**actual** effect on statistical power, rather than the original target.

### n = 3 (target)
Ground truth was established by three independent raters scoring all 28 pairs
blind against the shared anchored rubric. Each reconciled score is the mean of the
three judgments; inter-rater reliability is reported as the mean pairwise Pearson
correlation across the three rater pairs. The 28-pair set remains proof-of-concept
scale (PRD §7.3), so results are directional rather than definitive.

### n = 2
Ground truth was established by **two** independent raters rather than the targeted
three. Inter-rater agreement (mean pairwise Pearson) is computed from a single
rater pair; this **reduces, but does not eliminate,** the statistical power of the
agreement measure, and a single divergent pair influences the statistic more than
it would under three raters. Each reconciled score is the mean of two judgments.
This is disclosed as a deliberate methodology decision under Design Blueprint P3,
alongside the small-sample caveat of PRD §7.3.

### n = 1
Ground truth reflects a **single** rater's judgment. No inter-rater reliability
statistic can be computed as a result: `overall_inter_rater_agreement` is reported
as `null` and `n_raters` as `1`. This is a **genuine scope limitation, not a
footnote.** Single-rater ground truth cannot separate a systematic scoring bias
from true fit, and is **not equivalent in rigor** to the multi-rater ground truth
originally planned. It is disclosed here (P3) so that all downstream accuracy
figures are read with this constraint in view.

## 3. Reconciliation Method (unchanged across n)

| Output | Definition | n = 1 | n = 2 | n = 3 |
|---|---|---|---|---|
| `reconciled_score` | Mean of available rater scores | = the score | mean of 2 | mean of 3 |
| `inter_rater_range` | max − min across raters | `0.0` | computed | computed |
| `divergence_flag` | `range > 20.0` (provisional threshold) | never | possible | possible |
| `overall_inter_rater_agreement` | Mean pairwise Pearson over fully-rated pairs | `null` | 1 pair | 3 pairs |
| `n_raters` | Distinct raters present in the data | `1` | `2` | `3` |

Only pairs scored by **every** rater contribute to the overall Pearson metric, so a
partially rated pair cannot distort the reliability figure. Unrated pairs remain in
the `awaiting_raters` state and are never assigned a fabricated score.

## 4. Limitations & Roadmap (Report §14 §9)

**Limitations.** The evaluation rests on a proof-of-concept-scale ground truth: 28
curated resume/JD pairs scored by {N} human rater(s), consistent with the
small-data handling anticipated in PRD §7.3. This is sufficient to exercise the
confidence-band logic and to sanity-check ranking behavior across the clear-fit /
clear-gap / ambiguous strata, but is too small to support strong generalization
claims or tight confidence intervals on accuracy. {If n = 1: with a single rater,
no inter-rater reliability could be measured, so agreement between the model and a
*consensus* human judgment remains unverified.}

**Roadmap.** (1) Expand the ground-truth set toward and beyond the PRD's upper
bound; (2) recruit the full complement of independent raters to restore the
inter-rater reliability statistic; (3) re-tune the divergence threshold (currently
a provisional `20.0`) once real rater spread is observed.

## 5. Reproducibility Note

The reconciliation CLI accepts one or more `--sheets` arguments and was verified to
run without error for one- and two-sheet invocations, correctly holding the dataset
in the `awaiting_raters` state until real scores are present:

```
python -m scripts.reconcile_ground_truth \
    --curated data/ground_truth/curated.csv \
    --sheets data/ground_truth/rating_sheets/ratings_A.csv:A [ratings_B.csv:B ...]
```

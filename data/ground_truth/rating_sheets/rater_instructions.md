# Rater Instructions — HireLens Ground-Truth Scoring

Thank you for helping build HireLens's evaluation ground truth. Your job is to
judge how well each resume fits a specific job description. These are logistics
only — all scoring criteria live in the rubric, which you must read first.

Please follow these steps in order.

1. **What this is for.** You will score how well a candidate's resume matches a
   given job, so we can later measure how closely our system agrees with human
   judgment. Your scores are the human reference we compare against.

2. **Read the rubric first.** Open `RUBRIC.md` (one folder up from this file, in
   `data/ground_truth/`) and read it in full before you score anything. It defines
   what "fit" means and the 0–100 scale. Do not score from intuition alone — use
   the rubric's anchor points. This document does **not** repeat the criteria; the
   rubric is the single source of truth.

3. **Open your assigned sheet.** You have been assigned exactly one file in this
   `rating_sheets/` folder:
   - Rater A → `ratings_A.csv`
   - Rater B → `ratings_B.csv`
   - Rater C → `ratings_C.csv`

   Open **only** your own file. If you are unsure which one is yours, ask the
   project owner before opening anything.

4. **Fill in all 28 rows.** Each row is one resume/job pair. For every row, enter:
   - a **score** (0–100) in the `score` column, per the rubric's scale, and
   - a **justification** in the `justification` column — one short line naming the
     one or two factors that drove your score.

   Leave the `pair_id`, `resume_id`, and `jd_id` columns exactly as they are — do
   not edit, retype, or reformat them. (They are stored in a special text format so
   the long ID numbers don't get mangled; changing them breaks the link back to the
   source data.)

5. **Rate blind and independently.** Do **not** discuss your scores with the other
   raters, and do **not** look at anyone else's sheet, until the project owner tells
   you every rater has finished. Independent judgment is the whole point — comparing
   sheets early would invalidate the exercise.

6. **Save without renaming.** When done, save the file in place, keeping the **same
   filename** and **CSV format** (`.csv`). Do not "Save As" a new name, do not save
   as `.xlsx`, and do not move the file. If your spreadsheet program asks about
   keeping the CSV format on save, choose to keep it.

7. **Time estimate.** Plan for roughly **30–45 minutes** for all 28 rows. It's fine
   to take breaks — just save your progress.

8. **When you're done.** Notify the project owner that your sheet is complete. Once
   all assigned sheets are in, the team runs a reconciliation step that combines
   every rater's sheet into a single ground-truth dataset, averages the scores per
   pair, and measures how much the raters agreed. You don't need to do anything for
   that step — completing and saving your sheet is the finish line for you.

If anything is unclear, ask the project owner before entering scores rather than
guessing.

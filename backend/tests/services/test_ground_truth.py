"""Tests for the Phase 5.1 ground-truth construction infrastructure.

Rater scores here are SYNTHETIC unit-test fixtures used to verify the
reconciliation math — they are NOT, and must never be mistaken for, real
ground-truth ratings.
"""

from __future__ import annotations

from pathlib import Path

from app.services.evaluation.ground_truth_schema import (
    GroundTruthDataset,
    GroundTruthPair,
    RaterScore,
    load_dataset,
    save_dataset,
)
from app.services.evaluation.rater_tooling import (
    CuratedPair,
    assemble_pairs,
    load_rater_sheet,
    write_blind_rating_sheets,
)
from app.services.evaluation.reconciliation import (
    mean_pairwise_pearson,
    reconcile_dataset,
    reconcile_pair,
)


def _pair(pair_id: str, scores: dict[str, float], case="ambiguous") -> GroundTruthPair:
    return GroundTruthPair(
        pair_id=pair_id,
        resume_id=f"r-{pair_id}",
        jd_id=f"j-{pair_id}",
        case_type=case,  # type: ignore[arg-type]
        rater_scores=[
            RaterScore(rater_id=r, score=s, justification="x")
            for r, s in scores.items()
        ],
    )


def test_reconcile_pair_naive_mean_no_divergence() -> None:
    p = reconcile_pair(_pair("p1", {"A": 80, "B": 82, "C": 78}))
    assert p.reconciled_score == 80.0
    assert p.inter_rater_range == 4.0
    assert p.divergence_flag is False
    assert p.status == "reconciled"


def test_reconcile_pair_flags_sharp_divergence() -> None:
    p = reconcile_pair(_pair("p2", {"A": 80, "B": 20}))
    assert p.reconciled_score == 50.0
    assert p.inter_rater_range == 60.0
    assert p.divergence_flag is True  # 60 > threshold 20 → flagged, not hidden


def test_two_rater_case_is_handled() -> None:
    # Only 2 of a possible 3 raters available — must still reconcile + correlate.
    ds = GroundTruthDataset(
        pairs=[
            _pair("p1", {"A": 70, "B": 74}),
            _pair("p2", {"A": 40, "B": 44}),
            _pair("p3", {"A": 90, "B": 88}),
        ]
    )
    out = reconcile_dataset(ds)
    assert out.n_raters == 2
    assert out.overall_inter_rater_agreement is not None
    assert all(p.status == "reconciled" for p in out.pairs)


def test_perfectly_correlated_raters_agreement_is_one() -> None:
    # Rater B = rater A shifted by a constant → Pearson correlation exactly 1.0.
    agreement = mean_pairwise_pearson(
        {"A": [20.0, 50.0, 80.0], "B": [30.0, 60.0, 90.0]}
    )
    assert agreement == 1.0


def test_unrated_pair_never_gets_fabricated_score() -> None:
    ds = GroundTruthDataset(pairs=[_pair("p1", {})])  # no rater scores
    out = reconcile_dataset(ds)
    assert out.pairs[0].reconciled_score is None
    assert out.pairs[0].status == "awaiting_raters"


def test_blind_sheets_written_per_rater_and_ingest_roundtrip(tmp_path: Path) -> None:
    curated = [
        CuratedPair("p1", "r1", "j1", "clear_fit"),
        CuratedPair("p2", "r2", "j2", "clear_gap"),
    ]
    paths = write_blind_rating_sheets(curated, ["A", "B"], str(tmp_path))
    assert len(paths) == 2  # one blind sheet per rater
    # Blank sheets carry NO scores (nothing fabricated).
    assert load_rater_sheet(paths[0], "A") == {}

    # Simulate rater A filling their sheet (test-only synthetic input).
    (tmp_path / "ratings_A.csv").write_text(
        "pair_id,resume_id,jd_id,case_type,score,justification\n"
        "p1,r1,j1,clear_fit,85,strong\n"
        "p2,r2,j2,clear_gap,15,weak\n",
        encoding="utf-8",
    )
    sheet_a = load_rater_sheet(str(tmp_path / "ratings_A.csv"), "A")
    assert sheet_a["p1"].score == 85.0
    pairs = assemble_pairs(curated, {"A": sheet_a})
    assert pairs[0].rater_scores[0].rater_id == "A"


def test_dataset_save_load_roundtrip(tmp_path: Path) -> None:
    ds = reconcile_dataset(
        GroundTruthDataset(
            pairs=[_pair("p1", {"A": 80, "B": 82}), _pair("p2", {"A": 10, "B": 90})],
            notes="synthetic test fixture — NOT real ground truth",
        )
    )
    out = tmp_path / "gt.json"
    save_dataset(ds, str(out))
    reloaded = load_dataset(str(out))
    assert reloaded.model_dump() == ds.model_dump()
    # Divergent pair p2 kept its flag through persistence.
    p2 = next(p for p in reloaded.pairs if p.pair_id == "p2")
    assert p2.divergence_flag is True

"""Shared confidence banding utility.

``confidence_to_band`` is intentionally generic (float in → band out) so it is
reused for BOTH parsing_confidence (Phase 1.3) and scoring_confidence (Phase 6),
never duplicated. It lives here — separate from parsing_confidence.py — to make
that reuse explicit and discoverable.

Maps to: Design Blueprint §6.3 — this classification drives which visual state the
aperture-bloom score reveal renders.
"""

from __future__ import annotations

from app.schemas.scoring import ConfidenceLevel

# Thresholds (inclusive lower bounds). Reasoning:
# - HIGH ≥ 0.80: at least the two HIGH-weight signals plus supporting fields are
#   present — we can stand behind the result with a confident visual.
# - MEDIUM ≥ 0.50: a usable-but-caveated result; enough was extracted to be
#   meaningful, but visible caution is warranted.
# - LOW  < 0.50: too thin to trust silently — the UI must foreground the doubt
#   rather than present a confident-looking score (Design Blueprint P3).
HIGH_THRESHOLD = 0.80
MEDIUM_THRESHOLD = 0.50


def confidence_to_band(confidence: float) -> ConfidenceLevel:
    """Classify a [0.0, 1.0] confidence into a HIGH/MEDIUM/LOW band.

    Boundaries are inclusive at the lower edge: exactly 0.80 → HIGH, exactly
    0.50 → MEDIUM.
    """
    if confidence >= HIGH_THRESHOLD:
        return ConfidenceLevel.HIGH
    if confidence >= MEDIUM_THRESHOLD:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW

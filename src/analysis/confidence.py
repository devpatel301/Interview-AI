"""Heuristic confidence scoring engine."""

from __future__ import annotations


# Weights from TODO spec:
# confidence = 0.3*eye_contact + 0.2*posture + 0.3*speaking_consistency + 0.2*emotional_stability
_WEIGHTS = {
    "eye_contact": 0.30,
    "posture": 0.20,
    "speaking_consistency": 0.30,
    "emotional_stability": 0.20,
}


def compute_speaking_consistency(pause_ratio: float | None,
                                  speaking_ratio: float | None) -> float:
    """Convert speaking/pause stats to a speaking consistency score [0, 1].

    A pause_ratio near 0.15–0.25 is ideal (natural pauses).
    Very high pauses (> 0.6) or very low (< 0.05) score lower.
    """
    if pause_ratio is None or speaking_ratio is None:
        return 0.5  # neutral default when audio is unavailable

    # ideal pause band
    IDEAL_LOW, IDEAL_HIGH = 0.10, 0.30
    if IDEAL_LOW <= pause_ratio <= IDEAL_HIGH:
        return 1.0
    if pause_ratio < IDEAL_LOW:
        # speaking too fast / no pauses – mild penalty
        return max(0.4, 1.0 - (IDEAL_LOW - pause_ratio) * 5)
    # speaking too little
    return max(0.0, 1.0 - (pause_ratio - IDEAL_HIGH) * 2)


def compute_confidence_score(eye_contact_avg: float,
                              posture_avg: float,
                              speaking_consistency: float,
                              emotional_stability: float) -> float:
    """Weighted heuristic confidence score in [0, 1].

    Formula from TODO spec:
      confidence = 0.3*eye_contact + 0.2*posture + 0.3*speaking_consistency
                   + 0.2*emotional_stability
    """
    score = (
        _WEIGHTS["eye_contact"] * eye_contact_avg
        + _WEIGHTS["posture"] * posture_avg
        + _WEIGHTS["speaking_consistency"] * speaking_consistency
        + _WEIGHTS["emotional_stability"] * emotional_stability
    )
    return round(float(max(0.0, min(1.0, score))), 3)


def compute_engagement_score(eye_contact_avg: float,
                              movement_avg: float,
                              speaking_ratio: float | None) -> float:
    """Auxiliary engagement score in [0, 1].

    High eye contact + moderate movement + reasonable speaking time = engaged.
    """
    speaking_ratio = speaking_ratio if speaking_ratio is not None else 0.5
    # penalise extremes: no movement or too much movement
    movement_score = 1.0 - abs(movement_avg - 0.3) / 0.3 if movement_avg < 0.6 else 0.2
    movement_score = max(0.0, min(1.0, movement_score))

    score = 0.5 * eye_contact_avg + 0.3 * speaking_ratio + 0.2 * movement_score
    return round(float(max(0.0, min(1.0, score))), 3)


def confidence_label(score: float) -> str:
    """Return a human-readable label for a confidence score."""
    if score >= 0.80:
        return "High"
    if score >= 0.55:
        return "Moderate"
    if score >= 0.35:
        return "Low"
    return "Very Low"

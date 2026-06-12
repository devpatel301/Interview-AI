"""Emotion detection using DeepFace with a graceful fallback."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from deepface import DeepFace
    _DEEPFACE_AVAILABLE = True
except Exception:
    _DEEPFACE_AVAILABLE = False
    logger.warning("DeepFace is not available. Emotion detection will be disabled.")

# Canonical emotion labels
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Emotional stability weight: positive / calm emotions get a high weight
STABILITY_WEIGHTS = {
    "happy": 1.0,
    "neutral": 0.9,
    "surprise": 0.5,
    "sad": 0.3,
    "fear": 0.2,
    "angry": 0.1,
    "disgust": 0.0,
}


def analyse_frame_emotion(bgr_frame: np.ndarray,
                           enforce_detection: bool = False) -> Optional[dict]:
    """Run DeepFace emotion inference on a single BGR frame.

    Returns a dict with keys: dominant_emotion, emotion_scores (dict), or
    None if no face was detected or DeepFace is unavailable.
    """
    if not _DEEPFACE_AVAILABLE:
        return None

    try:
        result = DeepFace.analyze(
            bgr_frame,
            actions=["emotion"],
            enforce_detection=enforce_detection,
            silent=True,
        )
        if isinstance(result, list):
            result = result[0]

        dominant = result.get("dominant_emotion", "neutral")
        scores = result.get("emotion", {})
        # normalise scores to [0, 1]
        total = sum(scores.values()) or 1.0
        normalised = {k: round(v / total, 4) for k, v in scores.items()}

        return {
            "dominant_emotion": dominant,
            "emotion_scores": normalised,
        }
    except Exception as exc:
        logger.debug("DeepFace inference failed: %s", exc)
        return None


def compute_emotional_stability(emotion_timeline: list[dict]) -> float:
    """Compute an emotional stability score from the per-frame emotion results.

    Returns a value in [0, 1] where 1.0 means consistently calm / positive.
    Returns 0.5 if the timeline is empty or emotion data is missing.
    """
    weighted_scores: list[float] = []
    for entry in emotion_timeline:
        dominant = entry.get("dominant_emotion")
        if dominant:
            w = STABILITY_WEIGHTS.get(dominant, 0.5)
            weighted_scores.append(w)

    if not weighted_scores:
        return 0.5

    mean_score = float(np.mean(weighted_scores))
    # penalise high volatility (std dev of stability weights)
    volatility_penalty = float(np.std(weighted_scores)) * 0.3
    stability = max(0.0, min(1.0, mean_score - volatility_penalty))
    return round(stability, 3)


def emotion_summary(emotion_timeline: list[dict]) -> dict:
    """Aggregate emotion counts and percentages across a session."""
    counts: dict[str, int] = {label: 0 for label in EMOTION_LABELS}
    total = 0

    for entry in emotion_timeline:
        dominant = entry.get("dominant_emotion")
        if dominant and dominant in counts:
            counts[dominant] += 1
            total += 1

    if total == 0:
        return {"counts": counts, "percentages": {k: 0.0 for k in counts}}

    percentages = {k: round(v / total * 100, 1) for k, v in counts.items()}
    return {"counts": counts, "percentages": percentages}

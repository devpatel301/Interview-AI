"""Core video analysis pipeline — refactored to use modular analysis sub-package.

This module is the single entry point for processing a video file.  It
orchestrates frame-level inference (face, pose, emotion) and audio-level
analytics, then aggregates the results into a structured report dict.
"""

from __future__ import annotations

import os
import logging

import cv2
import numpy as np

from src.analysis.face import (
    create_face_mesh,
    detect_landmarks,
    estimate_eye_contact,
    detect_gaze_direction,
    draw_face_landmarks,
)
from src.analysis.pose import (
    create_pose_model,
    detect_pose,
    estimate_posture,
    estimate_movement,
    detect_posture_label,
    draw_pose_landmarks,
)
from src.analysis.audio import compute_speaking_metrics
from src.analysis.emotion import (
    analyse_frame_emotion,
    compute_emotional_stability,
    emotion_summary,
)
from src.analysis.confidence import (
    compute_speaking_consistency,
    compute_confidence_score,
    compute_engagement_score,
    confidence_label,
)

logger = logging.getLogger(__name__)

# Run DeepFace emotion analysis every N frames to keep processing fast
_EMOTION_FRAME_INTERVAL = 15


def process_video(video_path: str, output_path: str | None = None) -> dict:
    """Process an interview video and return a structured analytics report.

    Parameters
    ----------
    video_path:  Absolute path to the input video file.
    output_path: Optional path for the annotated output video.
                 Defaults to `<input_base>_processed.mp4`.

    Returns
    -------
    A dict with all analytics fields needed by the dashboard and API.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path is None:
        base, _ = os.path.splitext(video_path)
        output_path = f"{base}_processed.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    face_landmarker = create_face_mesh()
    pose_landmarker = create_pose_model()

    eye_scores: list[float] = []
    posture_scores: list[float] = []
    movement_scores: list[float] = []
    emotion_timeline: list[dict] = []

    frame_index = 0
    prev_pose_landmarks = None
    last_emotion: dict | None = None  # cache between emotion frames

    logger.info("Processing video: %s (%d frames @ %.1f fps)", video_path, frame_count, fps)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_index += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_lms = detect_landmarks(face_landmarker, rgb)
        pose_lms = detect_pose(pose_landmarker, rgb)

        # --- Face / Eye contact ---
        eye_score = 0.0
        gaze = "none"
        if face_lms is not None:
            eye_score = estimate_eye_contact(face_lms, frame.shape)
            gaze = detect_gaze_direction(face_lms, frame.shape)
            draw_face_landmarks(frame, face_lms)
        eye_scores.append(eye_score)

        # --- Pose / Posture ---
        posture_score = 0.0
        movement = 0.0
        posture_lbl = "N/A"
        if pose_lms is not None:
            posture_score = estimate_posture(pose_lms)
            movement = estimate_movement(prev_pose_landmarks, pose_lms)
            posture_lbl = detect_posture_label(posture_score)
            draw_pose_landmarks(frame, pose_lms)
            prev_pose_landmarks = pose_lms
        posture_scores.append(posture_score)
        movement_scores.append(movement)

        # --- Emotion (sampled every N frames) ---
        if frame_index % _EMOTION_FRAME_INTERVAL == 0:
            last_emotion = analyse_frame_emotion(frame)
        dominant_emotion = last_emotion["dominant_emotion"] if last_emotion else None

        emotion_entry = {
            "frame": frame_index,
            "timestamp_sec": round(frame_index / fps, 2),
            "eye_contact": round(eye_score, 3),
            "posture": round(posture_score, 3),
            "gaze": gaze,
            "posture_label": posture_lbl,
            "dominant_emotion": dominant_emotion,
        }
        emotion_timeline.append(emotion_entry)

        # --- HUD overlay ---
        _draw_hud(frame, eye_score, posture_score, dominant_emotion, posture_lbl)
        writer.write(frame)

    cap.release()
    writer.release()
    face_landmarker.close()
    pose_landmarker.close()

    # --- Audio analytics ---
    logger.info("Computing audio metrics…")
    speaking_metrics = compute_speaking_metrics(video_path)

    # --- Session-level aggregation ---
    eye_avg = float(np.mean(eye_scores)) if eye_scores else 0.0
    posture_avg = float(np.mean(posture_scores)) if posture_scores else 0.0
    movement_avg = float(np.mean(movement_scores)) if movement_scores else 0.0
    emotional_stability = compute_emotional_stability(emotion_timeline)
    em_summary = emotion_summary(emotion_timeline)
    dominant_overall = max(em_summary["counts"], key=em_summary["counts"].get) \
        if any(v > 0 for v in em_summary["counts"].values()) else "N/A"

    speaking_consistency = compute_speaking_consistency(
        speaking_metrics.get("pause_ratio"),
        speaking_metrics.get("speaking_ratio"),
    )
    confidence = compute_confidence_score(eye_avg, posture_avg,
                                          speaking_consistency, emotional_stability)
    engagement = compute_engagement_score(eye_avg, movement_avg,
                                          speaking_metrics.get("speaking_ratio"))
    duration = frame_count / fps if fps else 0.0

    analytics = {
        "duration_sec": round(duration, 2),
        "total_frames": frame_index,
        "eye_contact_avg": round(eye_avg, 3),
        "posture_avg": round(posture_avg, 3),
        "movement_avg": round(movement_avg, 3),
        "emotional_stability": emotional_stability,
        "dominant_emotion": dominant_overall,
        "emotion_summary": em_summary,
        "speaking_consistency": round(speaking_consistency, 3),
        "confidence_score": confidence,
        "confidence_label": confidence_label(confidence),
        "engagement_score": engagement,
        "speaking_metrics": speaking_metrics,
        "emotion_timeline": emotion_timeline,
        "processed_video": output_path,
    }

    logger.info(
        "Done. Confidence=%.2f  Eye=%.2f  Posture=%.2f",
        confidence, eye_avg, posture_avg,
    )
    return analytics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw_hud(frame: np.ndarray, eye_score: float, posture_score: float,
              emotion: str | None, posture_lbl: str) -> None:
    """Draw a semi-transparent HUD overlay on the frame in-place."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    green = (80, 220, 100)
    white = (230, 230, 230)
    yellow = (60, 210, 255)

    cv2.putText(frame, f"Eye Contact : {eye_score:.2f}", (10, 25),
                font, 0.55, green, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Posture     : {posture_score:.2f} ({posture_lbl})", (10, 50),
                font, 0.55, white, 1, cv2.LINE_AA)
    if emotion:
        cv2.putText(frame, f"Emotion     : {emotion}", (10, 75),
                    font, 0.55, yellow, 1, cv2.LINE_AA)

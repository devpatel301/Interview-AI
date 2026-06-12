"""Body pose and posture analysis — MediaPipe Tasks API (v0.10+)."""

from __future__ import annotations

import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
    drawing_utils,
    drawing_styles,
    PoseLandmark,
    PoseLandmarksConnections,
)

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "pose_landmarker_lite.task"
)


def create_pose_model() -> PoseLandmarker:
    """Return a PoseLandmarker configured for per-frame IMAGE mode."""
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


def detect_pose(landmarker: PoseLandmarker, rgb_frame: np.ndarray):
    """Run pose detection on an RGB numpy frame.

    Returns the first pose's NormalizedLandmark list, or None.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)
    if result.pose_landmarks:
        return result.pose_landmarks[0]
    return None


def estimate_posture(pose_landmarks) -> float:
    """Compute posture quality score [0, 1].

    Based on the spine proxy angle (shoulder centre → hip centre).
    """
    if pose_landmarks is None:
        return 0.0

    lm = pose_landmarks
    # PoseLandmark enum values map to indices
    LEFT_SHOULDER = PoseLandmark.LEFT_SHOULDER
    RIGHT_SHOULDER = PoseLandmark.RIGHT_SHOULDER
    LEFT_HIP = PoseLandmark.LEFT_HIP
    RIGHT_HIP = PoseLandmark.RIGHT_HIP

    left_shoulder = np.array([lm[LEFT_SHOULDER].x, lm[LEFT_SHOULDER].y])
    right_shoulder = np.array([lm[RIGHT_SHOULDER].x, lm[RIGHT_SHOULDER].y])
    left_hip = np.array([lm[LEFT_HIP].x, lm[LEFT_HIP].y])
    right_hip = np.array([lm[RIGHT_HIP].x, lm[RIGHT_HIP].y])
    left_ear = np.array([lm[PoseLandmark.LEFT_EAR].x, lm[PoseLandmark.LEFT_EAR].y])
    right_ear = np.array([lm[PoseLandmark.RIGHT_EAR].x, lm[PoseLandmark.RIGHT_EAR].y])

    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2
    ear_center = (left_ear + right_ear) / 2
    
    vector = hip_center - shoulder_center
    lean_angle = np.degrees(np.arctan2(vector[0], vector[1]))
    lean_score = 1.0 - min(abs(lean_angle) / 30.0, 1.0)

    torso_length = np.linalg.norm(hip_center - shoulder_center)
    neck_length = np.linalg.norm(shoulder_center - ear_center)
    neck_ratio = neck_length / (torso_length + 1e-6)
    
    slouch_score = np.clip((neck_ratio - 0.15) / 0.25, 0.0, 1.0)
    
    score = (lean_score * 0.6) + (slouch_score * 0.4)
    return float(np.clip(score, 0.0, 1.0))


def estimate_movement(prev_landmarks, curr_landmarks) -> float:
    """Compute normalised movement intensity between two consecutive frames."""
    if prev_landmarks is None or curr_landmarks is None:
        return 0.0

    prev_pts = np.array([[lm.x, lm.y] for lm in prev_landmarks])
    curr_pts = np.array([[lm.x, lm.y] for lm in curr_landmarks])
    movement = float(np.mean(np.linalg.norm(curr_pts - prev_pts, axis=1)))
    return float(np.clip(movement * 20.0, 0.0, 1.0))


def detect_posture_label(score: float) -> str:
    """Map a numeric posture score to a human-readable label."""
    if score >= 0.8:
        return "Upright"
    if score >= 0.55:
        return "Slightly Leaning"
    return "Slouching"


def draw_pose_landmarks(frame: np.ndarray, pose_landmarks) -> None:
    """Overlay body pose skeleton on a BGR frame in-place."""
    if pose_landmarks is None:
        return

    from mediapipe.framework.formats import landmark_pb2
    proto = landmark_pb2.NormalizedLandmarkList()
    for lm in pose_landmarks:
        proto.landmark.add(x=lm.x, y=lm.y, z=lm.z)

    drawing_utils.draw_landmarks(
        frame,
        proto,
        PoseLandmarksConnections.POSE_LANDMARKS,
        drawing_styles.get_default_pose_landmarks_style(),
    )

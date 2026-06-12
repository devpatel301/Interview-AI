"""Face mesh and eye contact analysis — MediaPipe Tasks API (v0.10+)."""

from __future__ import annotations

import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
    drawing_utils,
    drawing_styles,
    FaceLandmarksConnections,
)

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "face_landmarker.task"
)


def create_face_mesh() -> FaceLandmarker:
    """Return a FaceLandmarker configured for per-frame VIDEO mode."""
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return FaceLandmarker.create_from_options(options)


def detect_landmarks(landmarker: FaceLandmarker, rgb_frame: np.ndarray):
    """Run face landmark detection on an RGB numpy frame.

    Returns the first face's NormalizedLandmark list, or None.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)
    if result.face_landmarks:
        return result.face_landmarks[0]
    return None


def estimate_eye_contact(face_landmarks, image_shape) -> float:
    """Estimate eye contact score [0, 1] from FaceLandmarker landmarks.

    Checks how centred the mid-eye point is relative to the nose tip.
    """
    if face_landmarks is None:
        return 0.0

    h, w = image_shape[:2]
    points = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks])

    left_eye = points[33]
    right_eye = points[263]
    nose_tip = points[1]

    eye_center = (left_eye + right_eye) / 2
    distance = np.linalg.norm(eye_center - nose_tip)

    max_distance = w * 0.2
    score = max(0.0, 1.0 - distance / max_distance)
    return float(np.clip(score, 0.0, 1.0))


def detect_gaze_direction(face_landmarks, image_shape) -> str:
    """Return a coarse gaze label: 'center', 'left', 'right', 'up', 'down'."""
    if face_landmarks is None:
        return "none"

    h, w = image_shape[:2]
    points = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks])

    nose_tip = points[1]
    left_eye = points[33]
    right_eye = points[263]
    eye_center = (left_eye + right_eye) / 2

    dx = eye_center[0] - nose_tip[0]
    dy = eye_center[1] - nose_tip[1]

    if abs(dx) < w * 0.03 and abs(dy) < h * 0.04:
        return "center"
    if dx > w * 0.03:
        return "right"
    if dx < -w * 0.03:
        return "left"
    if dy < -h * 0.04:
        return "up"
    return "down"


def draw_face_landmarks(frame: np.ndarray, face_landmarks) -> None:
    """Overlay face contours on a BGR frame in-place using drawing_utils."""
    if face_landmarks is None:
        return

    # Build a proto-compatible landmark list for drawing_utils
    from mediapipe.framework.formats import landmark_pb2
    proto = landmark_pb2.NormalizedLandmarkList()
    for lm in face_landmarks:
        proto.landmark.add(x=lm.x, y=lm.y, z=lm.z)

    drawing_utils.draw_landmarks(
        frame,
        proto,
        FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
        drawing_styles.get_default_face_mesh_tesselation_style(),
    )
    drawing_utils.draw_landmarks(
        frame,
        proto,
        FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
        drawing_styles.get_default_face_mesh_contours_style(),
    )

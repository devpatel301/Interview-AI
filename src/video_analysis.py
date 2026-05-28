import os
import cv2
import numpy as np
import mediapipe as mp
from moviepy.editor import VideoFileClip

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def extract_audio_amplitude(video_path, fps=22050):
    """Extract the audio waveform from a video file.

    This function uses MoviePy to load the audio track and returns
    a normalized amplitude array and the sampling rate.
    """
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        return None, None

    audio = clip.audio.to_soundarray(fps=fps)
    clip.close()

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    amplitude = np.abs(audio)
    amplitude = amplitude / (np.max(amplitude) + 1e-8)
    return amplitude, fps


def detect_voice_segments(amplitude, fps, threshold=0.02, min_silence_sec=0.3):
    """Simple voice activity detection from normalized audio amplitude."""
    active = amplitude > threshold
    samples_per_segment = int(min_silence_sec * fps)

    if len(active) == 0:
        return []

    segments = []
    start = None
    for i, is_active in enumerate(active):
        if is_active and start is None:
            start = i
        elif not is_active and start is not None:
            if i - start >= samples_per_segment:
                segments.append((start / fps, i / fps))
            start = None
    if start is not None:
        segments.append((start / fps, len(active) / fps))
    return segments


def compute_speaking_metrics(video_path):
    """Compute speaking pace and pause/filler metrics from audio."""
    amplitude, fps = extract_audio_amplitude(video_path)
    if amplitude is None:
        return {
            "speaking_pace_wpm": None,
            "pause_ratio": None,
            "voice_segments": [],
            "audio_duration": 0,
        }

    segments = detect_voice_segments(amplitude, fps)
    duration = len(amplitude) / fps
    speaking_time = sum(end - start for start, end in segments)
    pause_time = duration - speaking_time
    pause_ratio = pause_time / duration if duration > 0 else None
    speaking_pace_wpm = len(segments) * 30 / (duration / 60 + 1e-8)

    return {
        "speaking_pace_wpm": round(speaking_pace_wpm, 1),
        "pause_ratio": round(pause_ratio, 2) if pause_ratio is not None else None,
        "voice_segments": segments,
        "audio_duration": round(duration, 2),
    }


def estimate_eye_contact(face_landmarks, image_shape):
    """Estimate eye contact from face mesh landmarks.

    This uses the location of the eyes relative to the nose tip.
    If the face is centered and both eyes are visible, we give a positive score.
    """
    if face_landmarks is None:
        return 0.0

    h, w = image_shape[:2]
    points = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark])
    left_eye = points[33]  # approximate left eye
    right_eye = points[263]  # approximate right eye
    nose_tip = points[1]

    eye_center = (left_eye + right_eye) / 2
    horizontal_offset = abs(eye_center[0] - nose_tip[0])
    vertical_offset = abs(eye_center[1] - nose_tip[1])
    distance = np.linalg.norm([horizontal_offset, vertical_offset])

    max_distance = w * 0.2
    score = max(0.0, 1.0 - distance / max_distance)
    return float(np.clip(score, 0.0, 1.0))


def estimate_posture(pose_landmarks):
    """Estimate posture quality from body pose landmarks."""
    if pose_landmarks is None:
        return 0.0

    left_shoulder = np.array([pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                              pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y])
    right_shoulder = np.array([pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                               pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
    left_hip = np.array([pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,
                         pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y])
    right_hip = np.array([pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x,
                          pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y])

    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2
    vector = hip_center - shoulder_center
    angle = np.degrees(np.arctan2(vector[0], vector[1]))
    score = 1.0 - min(abs(angle) / 45.0, 1.0)
    return float(np.clip(score, 0.0, 1.0))


def process_video(video_path, output_path=None):
    """Process a video to compute interview analytics and save an annotated output."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_processed{ext}"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                      max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
    pose = mp_pose.Pose(static_image_mode=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    eye_scores = []
    posture_scores = []
    emotion_timeline = []
    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_index += 1
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        face_results = face_mesh.process(image)
        pose_results = pose.process(image)
        image.flags.writeable = True

        eye_score = 0.0
        posture_score = 0.0

        if face_results.multi_face_landmarks:
            eye_score = estimate_eye_contact(face_results.multi_face_landmarks[0], image.shape)
            mp_drawing.draw_landmarks(frame, face_results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION,
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=1))

        if pose_results.pose_landmarks:
            posture_score = estimate_posture(pose_results.pose_landmarks)
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        eye_scores.append(eye_score)
        posture_scores.append(posture_score)
        emotion_timeline.append({
            "frame": frame_index,
            "eye_contact": eye_score,
            "posture": posture_score,
        })

        cv2.putText(frame, f"Eye Contact: {eye_score:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Posture: {posture_score:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        writer.write(frame)

    cap.release()
    writer.release()
    face_mesh.close()
    pose.close()

    speaking = compute_speaking_metrics(video_path)
    duration = frame_count / fps if fps else 0.0

    analytics = {
        "duration_sec": round(duration, 2),
        "eye_contact_avg": round(float(np.mean(eye_scores)) if eye_scores else 0.0, 2),
        "posture_avg": round(float(np.mean(posture_scores)) if posture_scores else 0.0, 2),
        "confidence_score": round(((np.mean(eye_scores) + np.mean(posture_scores)) / 2) if eye_scores and posture_scores else 0.0, 2),
        "emotion_timeline": emotion_timeline,
        "processed_video": output_path,
        "speaking_metrics": speaking,
    }

    return analytics

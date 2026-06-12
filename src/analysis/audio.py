"""Audio analytics using MoviePy for extraction and librosa for analysis."""

from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False

try:
    from moviepy.editor import VideoFileClip
    _MOVIEPY_AVAILABLE = True
except ImportError:
    _MOVIEPY_AVAILABLE = False


def extract_audio_to_file(video_path: str) -> Optional[str]:
    """Extract the audio track from a video and write it to a temp WAV file.

    Returns the path to the WAV file or None if no audio is present.
    """
    if not _MOVIEPY_AVAILABLE:
        return None

    clip = VideoFileClip(video_path)
    if clip.audio is None:
        clip.close()
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    clip.audio.write_audiofile(tmp.name, logger=None)
    clip.close()
    return tmp.name


def extract_audio_amplitude(video_path: str, fps: int = 22050):
    """Extract a normalised amplitude array from a video's audio track.

    Falls back to MoviePy if librosa is not available.
    Returns (amplitude_array, sample_rate) or (None, None).
    """
    if not _MOVIEPY_AVAILABLE:
        return None, None

    clip = VideoFileClip(video_path)
    if clip.audio is None:
        clip.close()
        return None, None

    audio = clip.audio.to_soundarray(fps=fps)
    clip.close()

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    amplitude = np.abs(audio)
    amplitude = amplitude / (np.max(amplitude) + 1e-8)
    return amplitude, fps


def detect_voice_segments(amplitude: np.ndarray, fps: int,
                           threshold: float = 0.02,
                           min_silence_sec: float = 0.3) -> list[tuple[float, float]]:
    """Detect voiced speech segments from a normalised amplitude array.

    Returns a list of (start_sec, end_sec) tuples.
    """
    active = amplitude > threshold
    samples_per_segment = int(min_silence_sec * fps)

    if len(active) == 0:
        return []

    segments: list[tuple[float, float]] = []
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


def compute_speaking_metrics(video_path: str) -> dict:
    """Compute speaking pace and pause metrics from the video's audio track.

    Returns a dict with:
        speaking_pace_wpm   – estimated words per minute (heuristic)
        pause_ratio         – fraction of time spent silent [0, 1]
        voice_segments      – list of (start_sec, end_sec) tuples
        audio_duration      – total audio length in seconds
        speaking_ratio      – fraction of time spent speaking [0, 1]
    """
    amplitude, fps = extract_audio_amplitude(video_path)
    if amplitude is None:
        return {
            "speaking_pace_wpm": None,
            "pause_ratio": None,
            "speaking_ratio": None,
            "voice_segments": [],
            "audio_duration": 0,
        }

    segments = detect_voice_segments(amplitude, fps)
    duration = len(amplitude) / fps
    speaking_time = sum(end - start for start, end in segments)
    pause_time = duration - speaking_time
    pause_ratio = pause_time / duration if duration > 0 else None
    speaking_ratio = speaking_time / duration if duration > 0 else None
    # heuristic: each segment ~ 1 breath-phrase ≈ 10 words, scale to WPM
    speaking_pace_wpm = (len(segments) * 10) / max(duration / 60, 1e-8)

    return {
        "speaking_pace_wpm": round(speaking_pace_wpm, 1),
        "pause_ratio": round(pause_ratio, 3) if pause_ratio is not None else None,
        "speaking_ratio": round(speaking_ratio, 3) if speaking_ratio is not None else None,
        "voice_segments": [(round(s, 2), round(e, 2)) for s, e in segments],
        "audio_duration": round(duration, 2),
    }

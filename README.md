# Real-Time Multimodal Interview Analytics Platform

## Project Overview

This project builds a lightweight, Docker-contained platform for analyzing interview videos and generating behavioral analytics. It is designed to support:

- uploaded interview recordings
- analysis of visual and audio signals
- structured metrics for eye contact, posture, confidence, speaking, and engagement
- a dashboard with annotated video and performance reports

## Core Problem Statement

Traditional interview feedback is subjective, inconsistent, and hard to review later. This platform introduces a reproducible analytics pipeline to measure behavioral signals during interviews and present them in a structured dashboard.

## What this MVP does

The current starter implementation includes:

- face and pose detection using MediaPipe
- eye contact and posture heuristics
- audio-based speaking / pause analytics using MoviePy
- a Streamlit dashboard for uploading video files
- Docker packaging for reproducible deployment

## Project Structure

- `app.py` – Streamlit application to upload video and show analytics.
- `src/video_analysis.py` – core video/audio processing and analytics extraction.
- `requirements.txt` – dependency list.
- `Dockerfile` – container build definition.
- `docker-compose.yml` – local compose orchestration for the app.
- `.dockerignore` – files excluded from Docker build.

## Getting Started

### Local Setup (Python)

1. Create a Python virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```bash
   streamlit run app.py
   ```
4. Open the browser URL shown in the terminal (usually `http://localhost:8501`).

### Docker Setup

1. Build the Docker image:
   ```bash
   docker build -t interview-ai:latest .
   ```
2. Run the container:
   ```bash
   docker run -p 8501:8501 interview-ai:latest
   ```
3. Or run with Docker Compose:
   ```bash
   docker compose up --build
   ```
4. Open `http://localhost:8501` in your browser.

## How it works

1. `app.py` provides a web interface and accepts uploaded video files.
2. `src/video_analysis.py` reads the video frames and audio track.
3. MediaPipe detects facial landmarks and body pose for each frame.
4. Simple heuristics compute:
   - eye contact score
   - posture score
   - confidence-like metric
   - voice activity segments
   - speaking pace and pause ratio
5. The system writes an annotated output video and displays summary analytics.

## Next Improvements

This setup is a strong foundation for a more professional platform. The next enhancements include:

- adding live webcam support using a browser media stream
- replacing heuristics with trained gaze, emotion, and posture models
- adding speech-to-text to compute actual words-per-minute and filler words
- storing analytics and reports in a database
- exporting PDF interview summaries
- adding authentication and multi-user session support

## Notes for Beginners

This project is meant for learning:
- You do not need an advanced ML background to start.
- The current code uses built-in computer vision tools and simple audio signal processing.
- Read `src/video_analysis.py` line-by-line to understand how the analytics are computed.
- Use Docker to package the app so it runs the same way on any machine.

Interview-AI is a beginner-friendly project for analyzing interview videos.
It uses computer vision and simple audio analysis to give feedback on:

- Eye contact
- Posture
- Speaking pace
- Pauses / filler detection
- Confidence estimation
- Engagement timeline
- Processed annotated video output

## Project Structure

- `app.py`: Streamlit dashboard for uploading video and showing results.
- `requirements.txt`: Python dependencies.
- `src/video_analysis.py`: Core analytics pipeline using OpenCV, MediaPipe, and MoviePy.

## Setup

1. Open a terminal in this repository.
2. Create a Python virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run the app

```bash
streamlit run app.py
```

Then open the browser page that Streamlit shows.

## How it works

- `app.py` provides a simple web interface.
- When you upload a video, `src/video_analysis.py` processes it.
- MediaPipe detects face and body posture in each video frame.
- A simple audio-based voice activity detector estimates speaking pace and pauses.
- The app saves a processed video with landmarks and overlay scores.

## Learning notes

- This project is intentionally simple so you can understand each part.
- `MediaPipe` is a library that detects faces, eyes, and body pose.
- `OpenCV` reads video frames and writes annotated video.
- `moviepy` extracts audio from the input video for speaking analysis.

## Next steps you can add

- Replace heuristic eye contact with a real gaze estimation model.
- Add speech-to-text to count words and calculate words-per-minute.
- Use a trained emotion recognition model for an emotion timeline.
- Add a webcam capture mode for live practice videos.
- Build a nicer dashboard with charts and PDF report export.

## Notes for beginners

This repository is a starting point. You do not need to know advanced ML yet:
- Start by running the app and uploading a short video.
- Read the code in `src/video_analysis.py` to see how each metric is computed.
- Use the results to learn where the model is strong and where it is simple.

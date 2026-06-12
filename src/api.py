"""FastAPI backend for Interview-AI.

Endpoints
---------
POST /upload-video         Accept a video upload, schedule background processing.
GET  /analysis/{id}        Poll for processing status / results.
GET  /report/{id}          Download the annotated output video.
GET  /sessions             List all past sessions.
GET  /health               Health check.
"""

from __future__ import annotations

import os
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from src.db import init_db, create_session, save_result, update_session_status, \
    get_session, get_result, list_sessions
from src.video_analysis import process_video

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPLOAD_DIR = os.path.join(ROOT_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="Interview-AI API",
    description="Multimodal interview analytics — face, pose, audio, emotion.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared DB connection (thread-safe via SQLite WAL mode)
_db = init_db()


# ─────────────────────────────────────────────────────────────────────────────
# Background task
# ─────────────────────────────────────────────────────────────────────────────

def _run_analysis(video_path: str, session_id: str) -> None:
    try:
        analytics = process_video(video_path)
        save_result(_db, session_id, analytics)
        update_session_status(_db, session_id, "done")
    except Exception as exc:
        update_session_status(_db, session_id, "error")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.2.0"}


@app.post("/upload-video", summary="Upload an interview video for analysis")
async def upload_video(background_tasks: BackgroundTasks,
                       file: UploadFile = File(...)):
    """Accept a video upload and schedule background processing.

    Returns a `session_id` that can be polled via `GET /analysis/{id}`.
    """
    if not (file.content_type or "").startswith("video"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a video")

    session_id = str(uuid.uuid4())
    filename = f"{session_id}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, filename)

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    create_session(_db, session_id, file.filename or filename, save_path)
    background_tasks.add_task(_run_analysis, save_path, session_id)

    return {"session_id": session_id, "status": "processing"}


@app.get("/analysis/{session_id}", summary="Get analysis status or results")
def get_analysis(session_id: str):
    session = get_session(_db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if session["status"] == "processing":
        return {"session_id": session_id, "status": "processing"}

    if session["status"] == "error":
        raise HTTPException(status_code=500, detail="Analysis failed for this session")

    result = get_result(_db, session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not available")

    return JSONResponse(content=result.get("analytics", {}))


@app.get("/report/{session_id}", summary="Download the annotated output video")
def get_report(session_id: str):
    result = get_result(_db, session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Report not available yet")

    video_path = result.get("processed_video") or \
        (result.get("analytics") or {}).get("processed_video")

    if video_path and os.path.exists(video_path):
        return FileResponse(
            path=video_path,
            media_type="video/mp4",
            filename=os.path.basename(video_path),
        )

    raise HTTPException(status_code=404, detail="Processed video not found")


@app.get("/sessions", summary="List all past sessions")
def get_sessions(limit: int = 20):
    sessions = list_sessions(_db, limit=limit)
    return {"sessions": sessions, "count": len(sessions)}

"""SQLite persistence layer for Interview-AI sessions.

Schema
------
sessions
    id          TEXT  PRIMARY KEY  (UUID)
    created_at  TEXT  (ISO-8601)
    filename    TEXT
    video_path  TEXT
    status      TEXT  ('processing' | 'done' | 'error')

results
    session_id       TEXT  REFERENCES sessions(id)
    duration_sec     REAL
    eye_contact_avg  REAL
    posture_avg      REAL
    confidence_score REAL
    engagement_score REAL
    speaking_pace    REAL
    pause_ratio      REAL
    speaking_ratio   REAL
    emotional_stability REAL
    dominant_emotion TEXT
    processed_video  TEXT
    report_json      TEXT  (full JSON blob)
    created_at       TEXT
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

# Default DB path next to this file's parent directory
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DB_PATH = os.path.join(_ROOT, "data", "sessions.db")


def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Initialise the database schema and return an open connection."""
    conn = _connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            created_at  TEXT NOT NULL,
            filename    TEXT,
            video_path  TEXT,
            status      TEXT NOT NULL DEFAULT 'processing'
        );

        CREATE TABLE IF NOT EXISTS results (
            session_id          TEXT PRIMARY KEY REFERENCES sessions(id),
            duration_sec        REAL,
            eye_contact_avg     REAL,
            posture_avg         REAL,
            confidence_score    REAL,
            engagement_score    REAL,
            speaking_pace       REAL,
            pause_ratio         REAL,
            speaking_ratio      REAL,
            emotional_stability REAL,
            dominant_emotion    TEXT,
            processed_video     TEXT,
            report_json         TEXT,
            created_at          TEXT NOT NULL
        );
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def create_session(conn: sqlite3.Connection, session_id: str,
                   filename: str, video_path: str) -> None:
    """Insert a new session record with status='processing'."""
    conn.execute(
        "INSERT INTO sessions (id, created_at, filename, video_path, status) "
        "VALUES (?, ?, ?, ?, 'processing')",
        (session_id, _now(), filename, video_path),
    )
    conn.commit()


def update_session_status(conn: sqlite3.Connection, session_id: str,
                          status: str) -> None:
    conn.execute("UPDATE sessions SET status=? WHERE id=?", (status, session_id))
    conn.commit()


def get_session(conn: sqlite3.Connection, session_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    return dict(row) if row else None


def list_sessions(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        "SELECT s.*, r.confidence_score, r.eye_contact_avg, r.posture_avg, "
        "       r.speaking_pace, r.dominant_emotion "
        "FROM sessions s "
        "LEFT JOIN results r ON r.session_id = s.id "
        "ORDER BY s.created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def save_result(conn: sqlite3.Connection, session_id: str,
                analytics: dict) -> None:
    """Persist the full analytics result for a session."""
    sm = analytics.get("speaking_metrics", {})
    conn.execute(
        """INSERT OR REPLACE INTO results
           (session_id, duration_sec, eye_contact_avg, posture_avg,
            confidence_score, engagement_score, speaking_pace, pause_ratio,
            speaking_ratio, emotional_stability, dominant_emotion,
            processed_video, report_json, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            session_id,
            analytics.get("duration_sec"),
            analytics.get("eye_contact_avg"),
            analytics.get("posture_avg"),
            analytics.get("confidence_score"),
            analytics.get("engagement_score"),
            sm.get("speaking_pace_wpm"),
            sm.get("pause_ratio"),
            sm.get("speaking_ratio"),
            analytics.get("emotional_stability"),
            analytics.get("dominant_emotion"),
            analytics.get("processed_video"),
            json.dumps(analytics),
            _now(),
        ),
    )
    conn.commit()


def get_result(conn: sqlite3.Connection, session_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM results WHERE session_id=?", (session_id,)
    ).fetchone()
    if not row:
        return None
    data = dict(row)
    if data.get("report_json"):
        data["analytics"] = json.loads(data["report_json"])
    return data


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

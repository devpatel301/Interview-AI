"""Interview-AI — Premium Streamlit Dashboard.

Provides a polished UI for:
  • uploading interview videos
  • tracking analysis progress
  • displaying analytics (confidence, eye contact, posture, emotion, audio)
  • browsing session history from SQLite
"""

from __future__ import annotations

import os
import tempfile
import uuid

import numpy as np
import pandas as pd
import streamlit as st

from src.video_analysis import process_video
from src.db import init_db, create_session, save_result, update_session_status, \
    list_sessions, get_result

# ─────────────────────────────────────────────────────────────────────────────
# Page config & global styles
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Interview-AI Analytics",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Design tokens ── */
:root {
    --bg:             #EDE4CF; /* Saturated paper-yellow */
    --surface:        #E8E0C0; /* Saturated paper-yellow variant */
    --surface-high:   #E0D7B4;
    --surface-raised: #D5CB9E;
    --border:         #1C1A14; /* Heavy black/dark ink border */
    --ink:            #1C1A14; /* Dark ink */
    --muted:          #544E3D;
    --primary:        #FF006E; /* Risograph magenta */
    --accent:         #00FFCC; /* Risograph cyan */
    --tape:           rgba(235, 225, 185, 0.75);
}

/* Halftone dot texture on the whole app */
.stApp {
    background-color: var(--bg);
    color: var(--ink);
}

/* Mixed font collisions */
html, body, [class*="css"] {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    color: var(--ink);
}

/* Headings: serif Times New Roman colliding with Helvetica body */
h1, h2, h3 {
    font-family: 'Times New Roman', Times, 'Georgia', serif !important;
    color: var(--ink) !important;
    font-weight: 900 !important;
    letter-spacing: -0.01em;
    text-transform: uppercase;
    text-wrap: balance;
    /* Risograph misregistration shadow */
    text-shadow: 2.5px 0 var(--primary), -2.5px 0 var(--accent);
}

h1 {
    font-size: 2.8rem !important;
    margin-bottom: 20px !important;
}

h2 {
    font-size: 2rem !important;
}

h3 {
    font-size: 1.5rem !important;
}

/* Captions / labels / metadata: Courier monospace colliding with serif/sans */
.stCaption, small, code, pre, label, div[data-testid="metric-container"] label {
    font-family: 'Courier New', Courier, monospace !important;
    color: var(--muted) !important;
    font-weight: bold !important;
}

/* Sidebar: heavy border, flat background */
section[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 4px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--ink) !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
    color: var(--primary) !important;
}

/* Metric cards: Flat, heavy solid borders, tape effect */
div[data-testid="metric-container"] {
    background-color: var(--surface-high) !important;
    border: 3px solid var(--border) !important;
    border-radius: 0px !important; /* No rounded corners */
    padding: 24px 16px !important;
    box-shadow: 5px 5px 0px var(--border) !important; /* Raw shadow */
    position: relative;
    overflow: visible !important;
    transition: none !important; /* Break smooth motion */
}

/* No rotation for metric cards */
div[data-testid="metric-container"]:nth-child(odd) {
    transform: none !important;
}
div[data-testid="metric-container"]:nth-child(even) {
    transform: none !important;
}

/* Tape effect on metric cards */
div[data-testid="metric-container"]::before {
    content: "";
    position: absolute;
    top: -12px;
    left: 25%;
    width: 60px;
    height: 18px;
    background-color: var(--tape);
    border-left: 1px dashed rgba(0,0,0,0.15);
    border-right: 1px dashed rgba(0,0,0,0.15);
    transform: none;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    z-index: 10;
}

div[data-testid="metric-container"] label {
    font-size: 0.8rem !important;
    text-transform: uppercase;
    color: var(--ink) !important;
}

/* Metric values: Courier, misregistered Risograph shadow */
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-size: 2.2rem !important;
    font-weight: 900 !important;
    color: var(--ink) !important;
    font-family: 'Courier New', Courier, monospace !important;
    text-shadow: 2px 0 var(--primary), -2px 0 var(--accent) !important;
    margin-top: 8px;
}

/* File Uploader: heavy border, flat background */
div[data-testid="stFileUploader"] {
    border: 3px dashed var(--border) !important;
    border-radius: 0px !important;
    padding: 24px !important;
    background-color: var(--surface-high) !important;
    box-shadow: -4px 4px 0px var(--border);
    transition: none !important;
}

/* Primary Button: heavy blocky style, flat, Risograph shadows */
button[kind="primary"], .stButton > button[kind="primaryFormSubmit"] {
    background-color: var(--primary) !important;
    color: #FFF !important;
    border: 3px solid var(--border) !important;
    border-radius: 0px !important;
    font-family: 'Times New Roman', Times, serif !important;
    font-size: 1.25rem !important;
    font-weight: 900 !important;
    text-transform: uppercase;
    padding: 12px 24px !important;
    box-shadow: 3px 3px 0px var(--border) !important;
    text-shadow: 1.5px 0 var(--accent);
    transition: none !important;
}
button[kind="primary"]:hover {
    background-color: var(--accent) !important;
    color: var(--border) !important;
    text-shadow: 1.5px 0 var(--primary);
    box-shadow: 1px 1px 0px var(--border) !important;
}
button[kind="primary"]:active {
    transform: translate(2px, 2px) !important;
    box-shadow: none !important;
}

/* Hover effect on File Uploader */
div[data-testid="stFileUploader"]:hover {
    background-color: var(--surface-raised) !important;
    border-color: var(--primary) !important;
}

/* Tabs: raw tabs, monospace font, flat container */
div[data-testid="stTabBar"] {
    border-bottom: 3px solid var(--border) !important;
}
button[data-baseweb="tab"] {
    font-family: 'Courier New', Courier, monospace !important;
    font-weight: 900 !important;
    color: var(--muted) !important;
    border-radius: 0px !important;
    border: 3px solid transparent !important;
    margin-right: 6px !important;
}
button[data-baseweb="tab"]:hover {
    color: var(--ink) !important;
    background-color: var(--tape) !important;
    border: 3px dashed var(--border) !important;
    border-bottom-color: var(--tape) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--ink) !important;
    background-color: var(--surface-high) !important;
    border: 3px solid var(--border) !important;
    border-bottom-color: var(--surface-high) !important;
}

/* Progress bar: raw thick line */
div[data-testid="stProgress"] {
    border: 3px solid var(--border);
    background-color: var(--surface-high);
    height: 24px !important;
}
div[data-testid="stProgress"] > div > div {
    background-color: var(--primary) !important;
    height: 18px !important;
}

/* Dataframe & Tables: Monospace text, solid dark grid */
div[data-testid="stDataFrame"] {
    border-radius: 0px;
    border: 3px solid var(--border);
    box-shadow: 4px 4px 0px var(--border);
}

/* Alerts: flat yellow paper, heavy border */
div[data-testid="stAlert"] {
    background-color: var(--surface) !important;
    border: 3px solid var(--border) !important;
    border-radius: 0px !important;
    color: var(--ink) !important;
    box-shadow: 3px 3px 0px var(--border);
}

/* Landing card style */
.landing-card {
    background-color: var(--surface-high);
    border: 3px solid var(--border);
    padding: 20px;
    height: 130px;
    position: relative;
    box-shadow: 4px 4px 0px var(--border);
}
.landing-card .card-title {
    font-family: 'Times New Roman', Times, serif;
    font-weight: 900;
    font-size: 1.15rem;
    text-transform: uppercase;
    color: var(--ink);
    margin-bottom: 6px;
    text-shadow: 1px 0 var(--primary);
}
.landing-card .card-desc {
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.8rem;
    color: var(--muted);
    line-height: 1.4;
}

/* Flat landing cards */
.rotate-0 { transform: none; }
.rotate-1 { transform: none; }
.rotate-2 { transform: none; }
.rotate-3 { transform: none; }

/* Tape overlay on cards */
.taped-card::before {
    content: "";
    position: absolute;
    top: -12px;
    left: 35%;
    width: 55px;
    height: 16px;
    background-color: var(--tape);
    border-left: 1px dashed rgba(0,0,0,0.15);
    border-right: 1px dashed rgba(0,0,0,0.15);
    transform: none;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

/* Staple overlay styling */
.stapled-card::after {
    content: "";
    position: absolute;
    top: -4px;
    left: 12px;
    width: 16px;
    height: 3px;
    background-color: #8C8C8C;
    border-bottom: 1px solid #333;
    transform: none;
}

/* Dividers: raw dashed lines */
hr {
    border: none !important;
    border-top: 3px dashed var(--border) !important;
    margin: 24px 0 !important;
}

/* Disabled transition/motion completely */
* {
    transition: none !important;
    animation: none !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DB connection (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_db():
    return init_db()

conn = get_db()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<div style='font-family:\"Google Sans\",sans-serif; font-size:1.1rem;"
        " font-weight:700; color:oklch(0.94 0.006 220); padding:4px 0'>Interview-AI</div>",
        unsafe_allow_html=True,
    )
    st.caption("Behavioral Analytics")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Analyze Video", "Session History"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**How it works**")
    st.markdown("""
1. Upload a recording (MP4, MOV, AVI, MKV)
2. Pipeline extracts eye contact, posture, voice and emotion signals
3. A composite **confidence score** is computed
4. Annotated video and full report saved to history
""")
    st.divider()
    st.caption("MediaPipe · DeepFace · MoviePy · FastAPI")

# ─────────────────────────────────────────────────────────────────────────────
# Helper components
# ─────────────────────────────────────────────────────────────────────────────

EMOTION_EMOJI = {
    "happy": "😊", "neutral": "😐", "sad": "😔",
    "angry": "😠", "fear": "😨", "surprise": "😲",
    "disgust": "🤢", "N/A": "❓",
}
CONFIDENCE_COLOR = {
    "High": "🟢", "Moderate": "🟡", "Low": "🟠", "Very Low": "🔴",
}


def _score_delta(score: float) -> str:
    """Return a descriptive delta string for st.metric."""
    if score >= 0.80:
        return "Excellent"
    if score >= 0.60:
        return "Good"
    if score >= 0.40:
        return "Fair"
    return "Needs work"


def render_score_ring(label: str, value: float, color: str = "#FF006E"):
    """Render a score as a circular SVG ring matching the paper-yellow zine style."""
    pct = int(value * 100)
    circumference = 2 * 3.14159 * 40
    dash = circumference * value
    gap = circumference - dash
    svg = f"""
    <div style="text-align:center; padding:8px">
      <svg width="96" height="96" viewBox="0 0 100 100" aria-label="{label}: {pct}%">
        <circle cx="50" cy="50" r="40" fill="none"
                stroke="#D0C7A4" stroke-width="8"/>
        <circle cx="50" cy="50" r="40" fill="none"
                stroke="{color}" stroke-width="8"
                stroke-dasharray="{dash:.1f} {gap:.1f}"
                stroke-linecap="square"
                transform="rotate(-90 50 50)"/>
        <text x="50" y="56" text-anchor="middle"
              font-size="18" font-weight="900" fill="#1C1A14"
              font-family="'Times New Roman', Times, serif"
              text-shadow="1px 0 #00FFCC">{pct}%</text>
      </svg>
      <div style="color:#544E3D; font-size:0.75rem;
                  font-weight:bold; letter-spacing:0.02em;
                  margin-top:4px; font-family:'Courier New',Courier,monospace">{label}</div>
    </div>"""
    return svg


# ─────────────────────────────────────────────────────────────────────────────
# Page: Analyze Video
# ─────────────────────────────────────────────────────────────────────────────

def page_analyze():
    st.title("Interview Analytics Platform")
    st.markdown(
        "<p style='color:var(--muted); font-size:1.05rem; margin-top:-8px; font-family:\"Courier New\", Courier, monospace; font-weight:bold'>"
        "Upload an interview recording and receive a full behavioral analytics report.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    uploaded = st.file_uploader(
        "Drag & drop your interview video here",
        type=["mp4", "mov", "avi", "mkv"],
        help="Supported formats: MP4, MOV, AVI, MKV",
    )

    if uploaded is None:
        _render_landing_info()
        return

    col_vid, col_info = st.columns([2, 1])
    with col_vid:
        st.video(uploaded)
    with col_info:
        st.markdown("#### 📄 File Details")
        st.markdown(f"""
| Field | Value |
|---|---|
| **Name** | `{uploaded.name}` |
| **Size** | `{uploaded.size / 1e6:.1f} MB` |
| **Type** | `{uploaded.type}` |
""")
        run = st.button("▶ Run Analysis", type="primary", use_container_width=True)

    if not run:
        return

    # Save to temp file
    session_id = str(uuid.uuid4())
    upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    video_filename = f"{session_id}_{uploaded.name}"
    video_path = os.path.join(upload_dir, video_filename)

    with open(video_path, "wb") as f:
        f.write(uploaded.getvalue())

    create_session(conn, session_id, uploaded.name, video_path)

    st.divider()
    progress_bar = st.progress(0, text="Initialising pipeline…")
    status_text = st.empty()

    try:
        status_text.info("🔍 Extracting frames and running MediaPipe…")
        progress_bar.progress(20, text="Running face & pose detection…")

        analytics = process_video(video_path)

        progress_bar.progress(80, text="Computing audio metrics…")
        status_text.info("🎵 Analysing audio…")

        save_result(conn, session_id, analytics)
        update_session_status(conn, session_id, "done")

        progress_bar.progress(100, text="Complete!")
        status_text.success("✅ Analysis complete!")

        _render_report(analytics, session_id)

    except Exception as exc:
        update_session_status(conn, session_id, "error")
        progress_bar.empty()
        status_text.error(f"❌ Processing failed: {exc}")
        st.exception(exc)


def _render_landing_info():
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Eye Contact",    "Gaze direction and camera-facing time per frame"),
        ("Posture",        "Body alignment quality and slouch detection"),
        ("Voice Analytics","Speaking pace, silence ratio and VAD segments"),
        ("Emotion",        "Per-frame DeepFace inference across the session"),
    ]
    for i, (col, (title, desc)) in enumerate(zip([c1, c2, c3, c4], cards)):
        rot_class = f"rotate-{i}"
        # Make some taped, some stapled
        card_effect = "taped-card" if i % 2 == 0 else "stapled-card"
        col.markdown(f"""
<div class="landing-card {rot_class} {card_effect}">
  <div class="card-title">{title}</div>
  <div class="card-desc">{desc}</div>
</div>""", unsafe_allow_html=True)


def _render_report(analytics: dict, session_id: str):
    st.divider()
    st.markdown("## 📊 Analytics Report")
    st.caption(f"Session ID: `{session_id}`")

    # ── Top score rings ──────────────────────────────────────────────────────
    r1, r2, r3, r4, r5 = st.columns(5)
    rings = [
        ("Confidence",     analytics.get("confidence_score", 0),  "#FF006E"),
        ("Eye Contact",    analytics.get("eye_contact_avg", 0),   "#00FFCC"),
        ("Posture",        analytics.get("posture_avg", 0),        "#FF006E"),
        ("Engagement",     analytics.get("engagement_score", 0),  "#00FFCC"),
        ("Emo Stability",  analytics.get("emotional_stability", 0), "#FF006E"),
    ]
    for col, (lbl, val, color) in zip([r1, r2, r3, r4, r5], rings):
        col.markdown(render_score_ring(lbl, val, color), unsafe_allow_html=True)

    st.divider()

    # ── Metric cards ─────────────────────────────────────────────────────────
    sm = analytics.get("speaking_metrics", {})
    conf = analytics.get("confidence_score", 0)
    conf_lbl = analytics.get("confidence_label", "N/A")
    emotion = analytics.get("dominant_emotion", "N/A")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("🎯 Confidence", f"{conf:.0%}", conf_lbl)
    m2.metric("👁️ Eye Contact", f"{analytics.get('eye_contact_avg', 0):.0%}",
              _score_delta(analytics.get('eye_contact_avg', 0)))
    m3.metric("🧍 Posture", f"{analytics.get('posture_avg', 0):.0%}",
              _score_delta(analytics.get('posture_avg', 0)))
    m4.metric("🎵 Speaking Pace", f"{sm.get('speaking_pace_wpm', 0):.0f} wpm")
    m5.metric("⏸️ Pause Ratio", f"{sm.get('pause_ratio', 0):.0%}" if sm.get('pause_ratio') is not None else "N/A")
    m6.metric("😊 Dominant Emotion", f"{EMOTION_EMOJI.get(emotion, '❓')} {emotion.title()}")

    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_timeline, tab_emotion, tab_audio, tab_video = st.tabs(
        ["📈 Timeline", "😊 Emotions", "🎵 Audio", "🎬 Annotated Video"]
    )

    with tab_timeline:
        _render_timeline_tab(analytics)

    with tab_emotion:
        _render_emotion_tab(analytics)

    with tab_audio:
        _render_audio_tab(analytics)

    with tab_video:
        _render_video_tab(analytics)


def _render_timeline_tab(analytics: dict):
    st.markdown("#### Per-Frame Analytics Timeline")
    timeline = analytics.get("emotion_timeline", [])
    if not timeline:
        st.warning("No timeline data available.")
        return

    df = pd.DataFrame(timeline)
    df_chart = df[["timestamp_sec", "eye_contact", "posture"]].set_index("timestamp_sec")
    df_chart.index.name = "Time (s)"
    df_chart.columns = ["Eye Contact", "Posture"]

    st.line_chart(df_chart, height=280, use_container_width=True)

    with st.expander("Raw timeline data (first 100 frames)"):
        display_cols = ["frame", "timestamp_sec", "eye_contact", "posture",
                        "gaze", "posture_label", "dominant_emotion"]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available].head(100), use_container_width=True)


def _render_emotion_tab(analytics: dict):
    st.markdown("#### Emotion Distribution")
    em_summary = analytics.get("emotion_summary", {})
    percentages = em_summary.get("percentages", {})

    if not percentages or all(v == 0 for v in percentages.values()):
        st.info("Emotion data not available — DeepFace may not be installed.")
        return

    em_df = pd.DataFrame({
        "Emotion": [f"{EMOTION_EMOJI.get(k, '')} {k.title()}" for k in percentages],
        "Percentage": list(percentages.values()),
    }).sort_values("Percentage", ascending=False)

    st.bar_chart(em_df.set_index("Emotion"), height=300, use_container_width=True)

    cols = st.columns(len(percentages))
    for col, (emotion, pct) in zip(cols, sorted(percentages.items(),
                                                  key=lambda x: x[1], reverse=True)):
        col.metric(
            f"{EMOTION_EMOJI.get(emotion, '❓')} {emotion.title()}",
            f"{pct:.1f}%",
        )


def _render_audio_tab(analytics: dict):
    sm = analytics.get("speaking_metrics", {})

    st.markdown("#### Speaking Analytics")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Duration", f"{analytics.get('duration_sec', 0):.1f}s")
    a2.metric("Speaking Pace", f"{sm.get('speaking_pace_wpm', 'N/A')} wpm")
    a3.metric("Pause Ratio", f"{sm.get('pause_ratio', 0):.1%}" if sm.get('pause_ratio') is not None else "N/A")
    a4.metric("Speaking Time", f"{sm.get('speaking_ratio', 0):.1%}" if sm.get('speaking_ratio') is not None else "N/A")

    voice_segs = sm.get("voice_segments", [])
    if voice_segs:
        st.markdown(f"#### Voice Activity Segments ({len(voice_segs)} segments detected)")
        seg_df = pd.DataFrame(voice_segs, columns=["Start (s)", "End (s)"])
        seg_df["Duration (s)"] = (seg_df["End (s)"] - seg_df["Start (s)"]).round(2)
        st.dataframe(seg_df, use_container_width=True)
    else:
        st.info("No voice segments detected or audio track missing.")


def _render_video_tab(analytics: dict):
    processed = analytics.get("processed_video")
    if processed and os.path.exists(processed):
        st.markdown("#### Annotated Video (with landmark overlays)")
        st.video(processed)
        with open(processed, "rb") as vf:
            st.download_button(
                "⬇ Download Annotated Video",
                data=vf,
                file_name=os.path.basename(processed),
                mime="video/mp4",
                use_container_width=True,
            )
    else:
        st.warning("Processed video not found.")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Session History
# ─────────────────────────────────────────────────────────────────────────────

def page_history():
    st.title("Session History")
    st.markdown(
        "<p style='color:var(--muted); font-family:\"Courier New\", Courier, monospace; font-weight:bold'>"
        "All past analysis sessions stored in SQLite.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    sessions = list_sessions(conn, limit=50)
    if not sessions:
        st.info("No sessions yet. Upload a video to get started.")
        return

    # Summary table
    rows = []
    for s in sessions:
        rows.append({
            "Session ID": s["id"][:8] + "…",
            "Filename": s.get("filename", "—"),
            "Status": s.get("status", "—"),
            "Confidence": f"{s['confidence_score']:.0%}" if s.get("confidence_score") is not None else "—",
            "Eye Contact": f"{s['eye_contact_avg']:.0%}" if s.get("eye_contact_avg") is not None else "—",
            "Posture": f"{s['posture_avg']:.0%}" if s.get("posture_avg") is not None else "—",
            "Emotion": s.get("dominant_emotion") or "—",
            "Created": s.get("created_at", "—")[:16].replace("T", " "),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Detail drill-down
    st.divider()
    st.markdown("#### 🔍 View Session Details")
    session_ids = [s["id"] for s in sessions]
    selected_id = st.selectbox("Select session", session_ids,
                                format_func=lambda x: x[:8] + "…")

    if selected_id:
        result = get_result(conn, selected_id)
        if result and result.get("analytics"):
            _render_report(result["analytics"], selected_id)
        else:
            st.warning("Result not yet available for this session.")


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

if "Analyze Video" in page:
    page_analyze()
else:
    page_history()

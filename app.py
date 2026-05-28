import os
import tempfile
import streamlit as st
import pandas as pd
from src.video_analysis import process_video

st.set_page_config(page_title="Real-Time Multimodal Interview Analytics Platform", layout="wide")

st.title("Real-Time Multimodal Interview Analytics Platform")
st.markdown(
    "Upload an interview recording and get visual, posture, and speaking analytics for interview feedback."
)

uploaded_file = st.file_uploader("Upload interview video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name

    st.info("Processing video, this may take a few minutes...")
    try:
        analytics = process_video(temp_video_path)

        st.success("Analysis complete!")
        st.video(analytics["processed_video"])

        st.header("Interview Analytics Report")
        summary = {
            "Video duration (s)": analytics["duration_sec"],
            "Eye contact score": analytics["eye_contact_avg"],
            "Posture score": analytics["posture_avg"],
            "Estimated confidence": analytics["confidence_score"],
            "Speaking pace (relative)": analytics["speaking_metrics"]["speaking_pace_wpm"],
            "Pause ratio": analytics["speaking_metrics"]["pause_ratio"],
        }
        st.write(pd.DataFrame(summary.items(), columns=["Metric", "Value"]))

        st.subheader("Voice activity")
        st.write(analytics["speaking_metrics"])

        st.subheader("Emotion / Engagement Timeline")
        timeline = pd.DataFrame(analytics["emotion_timeline"]).head(50)
        st.line_chart(timeline.set_index("frame")[['eye_contact', 'posture']])

        st.write("### Processed video saved to:")
        st.code(analytics["processed_video"])
    except Exception as exc:
        st.error(f"Processing failed: {exc}")

st.sidebar.header("Getting Started")
st.sidebar.write(
    "1. Install dependencies with `pip install -r requirements.txt`.\n"
    "2. Run `streamlit run app.py`.\n"
    "3. Upload a recorded interview video."
)

st.sidebar.write("Note: This starter app uses simple heuristics. It is designed for learning and can be extended with better ML models later.")

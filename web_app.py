import streamlit as st
import cv2
import numpy as np
import os
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer
import av
import time

# --- STYLING & UI ---
st.set_page_config(page_title="VisionAI Pro", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle, #1e293b 0%, #0f172a 100%);
        color: white;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(45deg, #3b82f6, #2563eb);
        color: white;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ VisionAI Professional Dashboard")
st.markdown("---")

# --- SETTINGS ---
st.sidebar.header("🕹️ AI Control Panel")
detector_choice = st.sidebar.selectbox(
    "Select Face Detector", 
    ["mtcnn", "retinaface", "yolov8"], 
    index=0,
    help="MTCNN (Speed), RetinaFace (Accuracy), YOLOv8 (Robustness)"
)

threshold = st.sidebar.slider("Match Sensitivity", 0.0, 1.0, 0.72)

DB_PATH = "faces"

# Model Warmup for Smoothness
@st.cache_resource
def warm_up_detector(detector_name):
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    try:
        DeepFace.extract_faces(img_path=dummy_img, detector_backend=detector_name, enforce_detection=False)
        return True
    except:
        return False

is_ready = warm_up_detector(detector_choice)

# Database Refresh
if st.sidebar.button("🔄 Sync Face Database"):
    for f in os.listdir(DB_PATH):
        if f.endswith(".pkl"):
            os.remove(os.path.join(DB_PATH, f))
    st.sidebar.success("Database Refreshed!")

# --- VIDEO CALLBACK ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    if not is_ready:
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    try:
        results = DeepFace.find(
            img_path=img, 
            db_path=DB_PATH, 
            model_name='ArcFace', 
            detector_backend=detector_choice, 
            enforce_detection=False,
            align=True, # Critical for MTCNN
            silent=True
        )
        
        if len(results) > 0 and not results[0].empty:
            match = results[0].iloc[0]
            dist = match['distance']
            confidence = max(0, min(100, (1 - dist/0.8) * 100))
            
            x, y, w, h = int(match['source_x']), int(match['source_y']), int(match['source_w']), int(match['source_h'])
            
            if dist < threshold:
                name = os.path.basename(match['identity']).split('.')[0]
                name = ''.join([i for i in name if not i.isdigit()]).capitalize()
                color, label = (0, 255, 0), f"{name} ({confidence:.1f}%)"
            else:
                color, label = (0, 0, 255), "Unknown"

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(img, (x, y - 30), (x + w, y), color, -1)
            cv2.putText(img, label, (x + 5, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    except:
        pass

    return av.VideoFrame.from_ndarray(img, format="bgr24")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("### 📡 Live Feed")
    webrtc_streamer(
        key="vision-ai-final", 
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )

with col2:
    st.markdown("### 📊 System Status")
    st.metric("Active Detector", detector_choice.upper())
    st.write("---")
    st.write("👥 **Recognized Database:**")
    known_files = [f for f in os.listdir(DB_PATH) if f.endswith(('.jpg', '.png'))]
    if known_files:
        names = set([''.join([i for i in f.split('.')[0] if not i.isdigit()]).capitalize() for f in known_files])
        for n in names: st.success(f"Verified: {n}")
    else: st.warning("Database empty.")

st.markdown("---")
st.caption("AI Face ID Pro | Version 4.1")

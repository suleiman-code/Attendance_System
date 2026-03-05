import streamlit as st
import cv2
import numpy as np
import os
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Page Config
st.set_page_config(page_title="AI Face Recognition", layout="wide")

st.title("🛡️ AI Face Recognition Dashboard")
st.markdown("---")

# Sidebar for Settings
st.sidebar.header("⚙️ Configuration")
detector_choice = st.sidebar.selectbox("Choose Detector", ["mtcnn", "opencv", "mediapipe"])
threshold = st.sidebar.slider("Match Threshold", 0.0, 1.0, 0.68)

DB_PATH = "faces"

# Ensure faces directory
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

# Database Refresh Logic
if st.sidebar.button("🔄 Refresh Database"):
    for f in os.listdir(DB_PATH):
        if f.endswith(".pkl"):
            os.remove(os.path.join(DB_PATH, f))
    st.sidebar.success("Database Refreshed!")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    try:
        results = DeepFace.find(img_path=img, 
                                db_path=DB_PATH, 
                                model_name='ArcFace', 
                                detector_backend=detector_choice, 
                                enforce_detection=False,
                                silent=True)
        
        if len(results) > 0 and not results[0].empty:
            match = results[0].iloc[0]
            dist = match['distance']
            confidence = max(0, min(100, (1 - dist/0.8) * 100))
            
            x, y, w, h = int(match['source_x']), int(match['source_y']), int(match['source_w']), int(match['source_h'])
            
            if dist < threshold:
                name = os.path.basename(match['identity']).split('.')[0]
                name = ''.join([i for i in name if not i.isdigit()]).capitalize()
                color = (0, 255, 0)
                label = f"{name} {confidence:.1f}%"
            else:
                color = (0, 0, 255)
                label = "Unknown"

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(img, (x, y - 35), (x + w, y), color, -1)
            cv2.putText(img, label, (x + 5, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    except Exception as e:
        pass

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Main Section
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎥 Live Feed")
    webrtc_streamer(key="face-rec", 
                    video_frame_callback=video_frame_callback,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

with col2:
    st.subheader("ℹ️ System Status")
    st.info(f"Active Model: ArcFace")
    st.info(f"Detector: {detector_choice.upper()}")
    
    # List known people
    st.write("👥 Known People in Database:")
    known_files = [f for f in os.listdir(DB_PATH) if f.endswith(('.jpg', '.png'))]
    if known_files:
        unique_names = list(set([''.join([i for i in f.split('.')[0] if not i.isdigit()]).capitalize() for f in known_files]))
        for name in unique_names:
            st.success(f"✓ {name}")
    else:
        st.warning("No faces found in database!")

st.markdown("---")
st.caption("Developed by Suleiman Code | Powered by DeepFace & Streamlit")

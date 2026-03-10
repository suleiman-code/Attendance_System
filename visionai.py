"""
🛡️ VISIONAI PRO - FACE RECOGNITION DASHBOARD
=============================================
Two Sections:
1. 📸 ENROLLMENT - Register new faces using camera
2. 🔍 RECOGNITION - Identify registered faces

Simple and smooth - uses Streamlit camera directly.
"""

import streamlit as st
import cv2
import numpy as np
import os
import csv
import pandas as pd
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="VisionAI Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    .big-title {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
    .success-box {
        background: linear-gradient(45deg, #11998e22, #38ef7d22);
        border: 1px solid #11998e;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(45deg, #764ba2, #667eea);
    }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
DB_PATH = "faces"
ATTENDANCE_FILE = "attendance_log.csv"
os.makedirs(DB_PATH, exist_ok=True)

# --- SESSION STATE ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'enrolled_today' not in st.session_state:
    st.session_state.enrolled_today = []
if 'recognized_today' not in st.session_state:
    st.session_state.recognized_today = set()

# --- HELPER FUNCTIONS ---
def get_registered_faces():
    """Get list of registered face names"""
    faces = set()
    for f in os.listdir(DB_PATH):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = ''.join([c for c in f.split('.')[0] if not c.isdigit()]).replace('_', ' ').strip().title()
            if name:
                faces.add(name)
    return sorted(list(faces))

def refresh_database():
    """Clear cached embeddings"""
    count = 0
    for f in os.listdir(DB_PATH):
        if f.endswith(".pkl"):
            os.remove(os.path.join(DB_PATH, f))
            count += 1
    return count

def log_attendance(name, confidence):
    """Log attendance to CSV"""
    today = datetime.now().strftime("%Y-%m-%d")
    key = f"{name}_{today}"
    
    if key not in st.session_state.recognized_today:
        st.session_state.recognized_today.add(key)
        
        if not os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(['Name', 'Date', 'Time', 'Confidence'])
        
        with open(ATTENDANCE_FILE, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                name, today,
                datetime.now().strftime("%H:%M:%S"),
                f"{confidence:.1f}%"
            ])
        return True
    return False

def get_attendance_data():
    """Read attendance CSV"""
    if os.path.exists(ATTENDANCE_FILE):
        return pd.read_csv(ATTENDANCE_FILE)
    return pd.DataFrame(columns=['Name', 'Date', 'Time', 'Confidence'])

def detect_face_haar(image):
    """Simple face detection using Haar Cascade"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, 1.1, 4)
    return faces

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    st.markdown("---")
    
    # Navigation
    st.markdown("### 📍 Navigation")
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = 'home'
    if st.button("📸 Enrollment", use_container_width=True):
        st.session_state.page = 'enrollment'
    if st.button("🔍 Recognition", use_container_width=True):
        st.session_state.page = 'recognition'
    if st.button("📊 Attendance Log", use_container_width=True):
        st.session_state.page = 'attendance'
    
    st.markdown("---")
    
    # Database info
    st.markdown("### 📁 Database")
    faces = get_registered_faces()
    st.info(f"**{len(faces)}** faces registered")
    
    if faces:
        with st.expander("👥 View All"):
            for f in faces:
                st.write(f"✅ {f}")
    
    if st.button("🔄 Refresh DB", use_container_width=True):
        count = refresh_database()
        st.success(f"Cleared {count} cache files!")
    
    st.markdown("---")
    st.caption("VisionAI Pro v2.0")

# --- MAIN CONTENT ---

# ==================== HOME PAGE ====================
if st.session_state.page == 'home':
    st.markdown('<p class="big-title">🛡️ VisionAI Pro</p>', unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color:#888;'>Advanced Face Recognition System</h4>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="section-card">
            <h3>📸 Face Enrollment</h3>
            <p>Register new faces in the system using your camera.</p>
            <ul>
                <li>Take multiple photos from different angles</li>
                <li>Good lighting improves accuracy</li>
                <li>Remove glasses for best results</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("➡️ Go to Enrollment", use_container_width=True, key="home_enroll"):
            st.session_state.page = 'enrollment'
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="section-card">
            <h3>🔍 Face Recognition</h3>
            <p>Identify registered faces and mark attendance.</p>
            <ul>
                <li>Real-time face detection</li>
                <li>Automatic attendance logging</li>
                <li>Confidence score display</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("➡️ Go to Recognition", use_container_width=True, key="home_recog"):
            st.session_state.page = 'recognition'
            st.rerun()
    
    # Quick stats
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    
    stat1, stat2, stat3 = st.columns(3)
    with stat1:
        st.metric("Registered Faces", len(get_registered_faces()))
    with stat2:
        df = get_attendance_data()
        today = datetime.now().strftime("%Y-%m-%d")
        today_count = len(df[df['Date'] == today]) if not df.empty else 0
        st.metric("Today's Attendance", today_count)
    with stat3:
        st.metric("Total Records", len(df))

# ==================== ENROLLMENT PAGE ====================
elif st.session_state.page == 'enrollment':
    st.markdown('<p class="big-title">📸 Face Enrollment</p>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#888;'>Register a new face in the system</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Name input
    name = st.text_input("👤 Enter Name", placeholder="e.g., Ahmed Khan")
    
    if name:
        name_clean = name.strip().replace(' ', '_').title()
        
        st.markdown("### 📷 Take Photo")
        st.info("💡 Tips: Look directly at camera, ensure good lighting, remove glasses")
        
        # Camera input
        camera_photo = st.camera_input("Click to capture your face")
        
        if camera_photo:
            # Convert to OpenCV format
            file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Detect face
            faces = detect_face_haar(image)
            
            if len(faces) > 0:
                st.success("✅ Face detected!")
                
                # Draw rectangle on face
                display_img = image.copy()
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Show preview
                st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), caption="Preview", use_container_width=True)
                
                # Save button
                if st.button("💾 Save Face", use_container_width=True, type="primary"):
                    timestamp = datetime.now().strftime("%H%M%S")
                    filename = f"{name_clean}_{timestamp}.jpg"
                    filepath = os.path.join(DB_PATH, filename)
                    
                    cv2.imwrite(filepath, image)
                    refresh_database()  # Clear cache for fresh recognition
                    
                    st.session_state.enrolled_today.append(name_clean)
                    st.success(f"✅ Saved: {filename}")
                    st.balloons()
            else:
                st.error("❌ No face detected! Please try again with better lighting.")
    else:
        st.warning("👆 Please enter a name first")
    
    # Show recently enrolled
    if st.session_state.enrolled_today:
        st.markdown("---")
        st.markdown("### ✅ Recently Enrolled")
        for n in st.session_state.enrolled_today[-5:]:
            st.write(f"• {n}")

# ==================== RECOGNITION PAGE ====================
elif st.session_state.page == 'recognition':
    st.markdown('<p class="big-title">🔍 Face Recognition</p>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#888;'>Identify faces and mark attendance</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Check if faces exist
    registered = get_registered_faces()
    if not registered:
        st.error("❌ No faces registered yet! Please enroll faces first.")
        if st.button("➡️ Go to Enrollment"):
            st.session_state.page = 'enrollment'
            st.rerun()
    else:
        st.success(f"✅ {len(registered)} faces in database: {', '.join(registered)}")
        
        st.markdown("### 📷 Capture Face to Recognize")
        
        # Camera input
        camera_photo = st.camera_input("Click to capture and recognize")
        
        if camera_photo:
            # Convert to OpenCV format
            file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            with st.spinner("🔄 Analyzing face..."):
                try:
                    from deepface import DeepFace
                    
                    results = DeepFace.find(
                        img_path=image,
                        db_path=DB_PATH,
                        model_name='ArcFace',
                        detector_backend='opencv',
                        enforce_detection=False,
                        silent=True
                    )
                    
                    display_img = image.copy()
                    found_someone = False
                    
                    for df in results:
                        if df.empty:
                            continue
                        
                        for idx, match in df.iterrows():
                            dist = match['distance']
                            confidence = max(0, min(100, (1 - dist/0.8) * 100))
                            
                            x = int(match['source_x'])
                            y = int(match['source_y'])
                            w = int(match['source_w'])
                            h = int(match['source_h'])
                            
                            if dist < 0.68:
                                identity = os.path.basename(match['identity']).split('.')[0]
                                identity = ''.join([c for c in identity if not c.isdigit()]).replace('_', ' ').strip().title()
                                
                                # Draw green box
                                cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                                label = f"{identity} ({confidence:.0f}%)"
                                cv2.putText(display_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                
                                found_someone = True
                                
                                # Log attendance
                                if log_attendance(identity, confidence):
                                    st.success(f"✅ **{identity}** identified! Attendance marked.")
                                else:
                                    st.info(f"ℹ️ **{identity}** already marked today.")
                            else:
                                # Draw red box
                                cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 0, 255), 3)
                                cv2.putText(display_img, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Show result
                    st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), caption="Recognition Result", use_container_width=True)
                    
                    if not found_someone:
                        st.warning("⚠️ No registered face found. Try enrolling first.")
                        
                except ImportError:
                    st.error("❌ DeepFace not installed! Run: `python -m pip install deepface tf-keras`")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Today's recognized
        st.markdown("---")
        st.markdown("### 📋 Today's Attendance")
        df = get_attendance_data()
        today = datetime.now().strftime("%Y-%m-%d")
        today_df = df[df['Date'] == today] if not df.empty else pd.DataFrame()
        
        if not today_df.empty:
            for _, row in today_df.iterrows():
                st.markdown(f"""
                <div class="success-box">
                    <strong>{row['Name']}</strong> — {row['Time']} ({row['Confidence']})
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No attendance recorded today yet.")

# ==================== ATTENDANCE PAGE ====================
elif st.session_state.page == 'attendance':
    st.markdown('<p class="big-title">📊 Attendance Log</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    df = get_attendance_data()
    
    if not df.empty:
        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Unique People", df['Name'].nunique())
        with col3:
            st.metric("Days Logged", df['Date'].nunique())
        
        st.markdown("---")
        
        # Filter by date
        dates = ['All'] + sorted(df['Date'].unique().tolist(), reverse=True)
        selected_date = st.selectbox("📅 Filter by Date", dates)
        
        if selected_date != 'All':
            df = df[df['Date'] == selected_date]
        
        # Show table
        st.dataframe(df.sort_values(['Date', 'Time'], ascending=[False, False]), 
                     use_container_width=True, hide_index=True)
        
        # Export
        st.markdown("---")
        csv_data = df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv_data,
            f"attendance_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            use_container_width=True
        )
    else:
        st.info("📭 No attendance records yet. Start recognizing faces to build the log!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#555; padding:20px;">
    🛡️ <strong>VisionAI Pro</strong> | Face Recognition & Attendance System<br>
    <small>Built with DeepFace, OpenCV & Streamlit</small>
</div>
""", unsafe_allow_html=True)

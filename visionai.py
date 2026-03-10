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
REQUIRED_PHOTOS = 6  # Minimum photos required for enrollment
os.makedirs(DB_PATH, exist_ok=True)

# Auto-refresh database cache on startup (once per session)
if 'db_refreshed' not in st.session_state:
    st.session_state.db_refreshed = True
    # Clear old cache files
    for f in os.listdir(DB_PATH):
        if f.endswith(".pkl"):
            os.remove(os.path.join(DB_PATH, f))

# --- SESSION STATE ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'enrolled_today' not in st.session_state:
    st.session_state.enrolled_today = []
if 'recognized_today' not in st.session_state:
    st.session_state.recognized_today = set()
    # Load today's attendance from CSV so state persists across sessions
    if os.path.exists(ATTENDANCE_FILE):
        today = datetime.now().strftime("%Y-%m-%d")
        try:
            existing_df = pd.read_csv(ATTENDANCE_FILE)
            for _, row in existing_df.iterrows():
                if str(row['Date']) == today:
                    st.session_state.recognized_today.add(f"{row['Name']}_{today}")
        except Exception:
            pass
if 'current_enrollment_name' not in st.session_state:
    st.session_state.current_enrollment_name = ''
if 'current_photo_count' not in st.session_state:
    st.session_state.current_photo_count = 0
if 'show_replace_dialog' not in st.session_state:
    st.session_state.show_replace_dialog = False
if 'temp_images' not in st.session_state:
    st.session_state.temp_images = []

# --- HELPER FUNCTIONS ---
def get_registered_faces():
    """Get list of registered face names - supports any naming format"""
    faces = set()
    for f in os.listdir(DB_PATH):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Support multiple naming formats:
            # Format 1: name_1.jpg, name_2.jpg (structured)
            # Format 2: name.jpg (simple)
            # Format 3: name_anything.jpg (manual add)
            basename = f.rsplit('.', 1)[0]  # Remove extension
            
            # Try to extract name (first part before underscore or full name)
            if '_' in basename:
                name = basename.split('_')[0]
            else:
                name = basename
            
            name = name.strip().title()
            if name and len(name) > 1:
                faces.add(name)
    return sorted(list(faces))

def get_user_photo_count(name):
    """Count how many photos exist for a user - supports any naming format"""
    name_lower = name.lower().replace(' ', '_')
    count = 0
    for f in os.listdir(DB_PATH):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            basename = f.rsplit('.', 1)[0].lower()
            # Match: name_1, name_2, name_anything, or just name
            if basename == name_lower or basename.startswith(name_lower + '_'):
                count += 1
    return count

def is_user_enrolled(name):
    """Check if user is already enrolled with 6 photos"""
    return get_user_photo_count(name) >= REQUIRED_PHOTOS

def is_user_registered(name):
    """Check if user has ANY photos in database (for recognition)"""
    return get_user_photo_count(name) >= 1

def get_next_photo_number(name):
    """Get next photo number for user (1-6)"""
    name_lower = name.lower().replace(' ', '_')
    existing_numbers = []
    for f in os.listdir(DB_PATH):
        if f.lower().startswith(name_lower + '_') and f.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                num = int(f.split('_')[1].split('.')[0])
                existing_numbers.append(num)
            except:
                pass
    
    for i in range(1, REQUIRED_PHOTOS + 1):
        if i not in existing_numbers:
            return i
    return None

def delete_user_photos(name):
    """Delete all photos for a user"""
    name_lower = name.lower().replace(' ', '_')
    deleted = 0
    for f in os.listdir(DB_PATH):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            basename = f.rsplit('.', 1)[0].lower()
            if basename == name_lower or basename.startswith(name_lower + '_'):
                os.remove(os.path.join(DB_PATH, f))
                deleted += 1
    refresh_database()
    return deleted

def is_attendance_marked_today(name):
    """Check if attendance already marked today for this person"""
    today = datetime.now().strftime("%Y-%m-%d")
    key = f"{name}_{today}"
    return key in st.session_state.recognized_today

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

def detect_face_advanced(image):
    """Advanced face detection - works with masks and angles"""
    faces = []
    
    # Try multiple detection methods
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Frontal face
    frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    frontal_faces = frontal_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    
    # Method 2: Frontal face alt (better for some angles)
    frontal_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    alt_faces = frontal_alt.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    
    # Method 3: Profile face (side view)
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    profile_faces = profile_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    
    # Combine all detections
    all_faces = list(frontal_faces) + list(alt_faces) + list(profile_faces)
    
    if len(all_faces) > 0:
        # Return the largest face found
        all_faces = sorted(all_faces, key=lambda f: f[2] * f[3], reverse=True)
        return [all_faces[0]]
    
    return []

def crop_face(image, face_coords, padding=30):
    """Crop only the face from image with padding"""
    x, y, w, h = face_coords
    height, width = image.shape[:2]
    
    # Add padding around face
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)
    
    # Crop face region
    face_img = image[y1:y2, x1:x2]
    
    # Resize to standard size for better recognition
    face_img = cv2.resize(face_img, (224, 224))
    
    return face_img

def check_liveness(image, face_coords):
    """
    Improved liveness detection - balanced to not reject real faces.
    Uses multiple checks with relaxed thresholds.
    Returns: (is_live, score, reason)
    """
    x, y, w, h = face_coords
    height, width = image.shape[:2]
    
    # Extract face region with padding
    pad = 20
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + w + pad)
    y2 = min(height, y + h + pad)
    face_region = image[y1:y2, x1:x2]
    
    if face_region.size == 0:
        return True, 100, "No face region - allowing"  # Don't block if can't analyze
    
    # Resize for consistent analysis
    try:
        face_region = cv2.resize(face_region, (200, 200))
    except:
        return True, 100, "Resize failed - allowing"
    
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    
    # CHECK 1: Laplacian variance (texture sharpness)
    laplacian = cv2.Laplacian(gray_face, cv2.CV_64F)
    laplacian_var = laplacian.var()
    
    # CHECK 2: Color variation
    hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
    s_std = np.std(hsv[:,:,1])  # Saturation variation only
    
    # CHECK 3: High frequency content (screens have less)
    dft = cv2.dft(np.float32(gray_face), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
    high_freq = np.mean(magnitude[75:125, 75:125])  # Center high frequencies
    
    score = 0
    reasons = []
    
    # Very relaxed thresholds - only catch obvious fakes
    # Real webcam faces typically: laplacian 20-500, s_std 15-60
    if laplacian_var > 15:
        score += 40
    else:
        reasons.append(f"Blur ({laplacian_var:.0f})")
    
    if s_std > 12:
        score += 35
    else:
        reasons.append(f"Flat ({s_std:.0f})")
    
    if high_freq > 1000:
        score += 25
    else:
        reasons.append(f"Screen pattern")
    
    # Pass at 40+ (was 50) - very lenient for real faces
    is_live = score >= 40
    reason = "LIVE" if is_live else "; ".join(reasons) if reasons else "Low quality"
    
    return is_live, score, reason

def get_existing_photo_numbers(name):
    """Get list of existing photo numbers for a user"""
    name_lower = name.lower()
    existing = []
    for f in os.listdir(DB_PATH):
        if f.lower().startswith(name_lower + '_') and f.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                num = int(f.split('_')[1].split('.')[0])
                existing.append(num)
            except:
                pass
    return sorted(existing)

def get_missing_photo_numbers(name):
    """Get list of missing photo numbers (1-6) for a user"""
    existing = get_existing_photo_numbers(name)
    missing = [i for i in range(1, REQUIRED_PHOTOS + 1) if i not in existing]
    return missing

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
    
    if st.button("🗑️ Clear Attendance", use_container_width=True):
        # Reset session state completely
        st.session_state.recognized_today = set()
        # Delete the attendance file
        if os.path.exists(ATTENDANCE_FILE):
            os.remove(ATTENDANCE_FILE)
        st.success("✅ All attendance records deleted!")
        st.rerun()
    
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
    
    # Quick stats - only registered faces
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    
    st.metric("Registered Users", len(get_registered_faces()))

# ==================== ENROLLMENT PAGE ====================
elif st.session_state.page == 'enrollment':
    st.markdown('<p class="big-title">📸 Face Enrollment</p>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#888;'>Register a new face in the system (6 photos required)</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Name input
    name = st.text_input("👤 Enter Name", placeholder="e.g., Suleiman")
    
    if name:
        name_clean = name.strip().replace(' ', '_').title()
        current_count = get_user_photo_count(name_clean)
        existing_nums = get_existing_photo_numbers(name_clean)
        missing_nums = get_missing_photo_numbers(name_clean)
        
        # CASE 1: User is fully enrolled (6 photos) - Show replace option
        if current_count >= REQUIRED_PHOTOS:
            st.success(f"✅ **{name_clean}** is already enrolled with {current_count} photos!")
            st.warning("⚠️ This user is already registered. Do you want to replace?")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, Replace All", use_container_width=True, type="primary"):
                    delete_user_photos(name_clean)
                    st.success(f"✅ All photos deleted. You can now re-enroll {name_clean}")
                    st.rerun()
            with col2:
                if st.button("❌ No, Keep", use_container_width=True):
                    st.info("Photos kept. Choose a different name or go to Recognition.")
        
        # CASE 2: Partial enrollment - Continue from where left
        elif current_count > 0:
            st.markdown(f"### 📊 Enrollment Progress: {current_count}/{REQUIRED_PHOTOS}")
            st.progress(current_count / REQUIRED_PHOTOS)
            st.warning(f"⚠️ **{name_clean}** has {current_count}/{REQUIRED_PHOTOS} photos - Continue enrollment")
            st.info(f"📸 Need to capture: Photo {', '.join([str(n) for n in missing_nums])}")
        
        # CASE 3: New user - First time enrollment
        else:
            st.markdown(f"### 📊 New Enrollment for {name_clean}")
            st.progress(0.0)
            st.info(f"👋 Welcome **{name_clean}**! Let's capture 6 photos for enrollment.")
        
        # Show enrollment form if not fully enrolled
        if current_count < REQUIRED_PHOTOS:
            next_photo_num = missing_nums[0] if missing_nums else current_count + 1
            remaining = REQUIRED_PHOTOS - current_count
            
            st.markdown("---")
            st.markdown(f"### 📷 Capture Photo #{next_photo_num} ({remaining} remaining)")
            st.markdown("""
            💡 **Tips for best results:**
            - 📷 Photo 1-2: Look **straight** at camera
            - 👈 Photo 3-4: **Turn left/right** slightly  
            - 😊 Photo 5-6: Different **expressions** or **with mask**
            - ☀️ Ensure **good lighting**
            """)
            
            # Camera input
            camera_photo = st.camera_input(f"📷 Capture Photo {next_photo_num} of {REQUIRED_PHOTOS} ({remaining} remaining)")
            
            if camera_photo:
                # Convert to OpenCV format
                file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Detect face using advanced detection (works with masks & angles)
                faces = detect_face_advanced(image)
                
                if len(faces) > 0:
                    st.success("✅ Face detected!")
                    
                    # Get face coordinates
                    (x, y, w, h) = faces[0]
                    
                    # Crop only face
                    face_only = crop_face(image, (x, y, w, h), padding=40)
                    
                    # Draw rectangle on original for preview
                    display_img = image.copy()
                    cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    
                    # Show both preview
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), caption="📷 Original (Face Detected)", use_container_width=True)
                    with col2:
                        st.image(cv2.cvtColor(face_only, cv2.COLOR_BGR2RGB), caption="✂️ Cropped Face (This will be saved)", use_container_width=True)
                    
                    # Save button
                    if st.button(f"💾 Save Face Photo {next_photo_num}", use_container_width=True, type="primary"):
                        filename = f"{name_clean}_{next_photo_num}.jpg"
                        filepath = os.path.join(DB_PATH, filename)
                        
                        # Save cropped face only
                        cv2.imwrite(filepath, face_only)
                        refresh_database()
                        
                        st.success(f"✅ Saved: {filename}")
                        
                        new_count = get_user_photo_count(name_clean)
                        if new_count >= REQUIRED_PHOTOS:
                            st.balloons()
                            st.success(f"🎉 **{name_clean}** successfully enrolled with {REQUIRED_PHOTOS} photos!")
                            st.session_state.enrolled_today.append(name_clean)
                        else:
                            new_remaining = REQUIRED_PHOTOS - new_count
                            st.info(f"📸 {new_remaining} more photo(s) to go!")
                        
                        st.rerun()
                else:
                    st.error("❌ No face detected! Try better lighting or different angle.")
        
        # Show enrolled photos if any exist
        if current_count > 0:
            st.markdown("---")
            st.markdown(f"### 📂 {name_clean}'s Enrolled Photos ({current_count}/{REQUIRED_PHOTOS})")
            cols = st.columns(6)
            for i in range(1, REQUIRED_PHOTOS + 1):
                filepath = os.path.join(DB_PATH, f"{name_clean}_{i}.jpg")
                with cols[i-1]:
                    if os.path.exists(filepath):
                        img = cv2.imread(filepath)
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"#{i} ✅", use_container_width=True)
                    else:
                        st.markdown(f"<div style='background:#333;padding:20px;text-align:center;border-radius:10px;'>#{i} ❌</div>", unsafe_allow_html=True)
    else:
        st.warning("👆 Please enter a name first")
    
    # Show registered users
    st.markdown("---")
    st.markdown("### 👥 All Registered Users")
    registered = get_registered_faces()
    if registered:
        for user in registered:
            count = get_user_photo_count(user)
            if count >= REQUIRED_PHOTOS:
                st.write(f"✅ **{user}** - {count} photos (Complete)")
            else:
                st.write(f"⏳ **{user}** - {count}/{REQUIRED_PHOTOS} photos (Incomplete)")
    else:
        st.info("No users registered yet.")

# ==================== RECOGNITION PAGE ====================
elif st.session_state.page == 'recognition':
    st.markdown('<p class="big-title">🔍 Real-Time Face Recognition</p>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#888;'>Live detection - Name & Score shown instantly!</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Check if faces exist
    registered = get_registered_faces()
    if not registered:
        st.error("❌ No faces registered yet! Please enroll faces first.")
        if st.button("➡️ Go to Enrollment"):
            st.session_state.page = 'enrollment'
            st.rerun()
    else:
        st.success(f"✅ {len(registered)} users in database: {', '.join(registered)}")
        
        st.markdown("### 📹 Live Camera Feed")
        st.info("💡 Camera detects face automatically - Name & Score shown in real-time!")
        
        # Camera input with auto-capture
        camera_photo = st.camera_input("📷 Point camera at face - Detection is automatic", key="recognition_camera")
        
        if camera_photo:
            # Convert to OpenCV format
            file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            display_img = image.copy()
            found_someone = False
            recognized_name = None
            recognized_confidence = 0
            already_marked = False
            spoof_detected = False
            
            try:
                from deepface import DeepFace
                
                # Use DeepFace directly on full image - it handles face detection
                results = DeepFace.find(
                    img_path=image,
                    db_path=DB_PATH,
                    model_name='ArcFace',
                    detector_backend='opencv',
                    enforce_detection=False,
                    silent=True
                )
                
                if results and len(results) > 0:
                    for df in results:
                        if df.empty:
                            continue
                        
                        # Sort by distance to get top matches
                        df_sorted = df.sort_values('distance')
                        best_match = df_sorted.iloc[0]
                        best_dist = best_match['distance']
                        
                        # Get face coordinates
                        fx = int(best_match['source_x'])
                        fy = int(best_match['source_y'])
                        fw = int(best_match['source_w'])
                        fh = int(best_match['source_h'])
                        
                        # Extract name from best match
                        fname = os.path.basename(best_match['identity'])
                        if '_' in fname:
                            identity = fname.split('_')[0].title()
                        else:
                            identity = fname.rsplit('.', 1)[0].title()
                        
                        # LIVENESS CHECK - Prevent photo spoofing
                        is_live, liveness_score, liveness_reason = check_liveness(image, (fx, fy, fw, fh))
                        
                        # IMPROVED RECOGNITION LOGIC
                        # 1. Check best match distance
                        # 2. Verify consistency - multiple photos of SAME person should match
                        # 3. Calculate gap between best match and other persons' matches
                        
                        THRESHOLD = 0.55  # Main threshold for recognition
                        
                        # Count how many photos of the identified person are in top matches
                        identity_lower = identity.lower()
                        same_person_matches = 0
                        other_person_best = 1.0  # Track best distance of OTHER people
                        
                        for _, row in df_sorted.head(6).iterrows():
                            row_fname = os.path.basename(row['identity'])
                            if '_' in row_fname:
                                row_identity = row_fname.split('_')[0].lower()
                            else:
                                row_identity = row_fname.rsplit('.', 1)[0].lower()
                            
                            if row_identity == identity_lower:
                                same_person_matches += 1
                            else:
                                # Track closest match of other people
                                if row['distance'] < other_person_best:
                                    other_person_best = row['distance']
                        
                        # Gap between best match and other person's best
                        gap = other_person_best - best_dist
                        
                        # Show debug info
                        st.markdown("#### 🔍 Recognition Analysis:")
                        debug_info = [
                            f"Best match: {identity} (distance: {best_dist:.3f})",
                            f"Same person in top 6: {same_person_matches} matches",
                            f"Gap to others: {gap:.3f}",
                            f"Liveness: {liveness_score}/100 ({'PASS' if is_live else 'FAIL'})"
                        ]
                        st.code("\n".join(debug_info))
                        
                        if not is_live:
                            st.warning(f"⚠️ Liveness issue: {liveness_reason}")
                        
                        # DECISION LOGIC:
                        # Registered person: low distance + multiple same-person matches
                        # Unregistered person: high distance OR few same-person matches OR small gap
                        
                        is_verified = False
                        reject_reason = ""
                        
                        if best_dist < THRESHOLD:
                            if is_user_registered(identity):
                                if is_live:
                                    # Additional check: consistency
                                    # Real person should have 2+ photos matching well
                                    if same_person_matches >= 2 or gap > 0.05:
                                        is_verified = True
                                        confidence = max(0, min(100, (1 - best_dist/0.6) * 100))
                                    else:
                                        reject_reason = "Inconsistent match - possible wrong person"
                                else:
                                    reject_reason = "Liveness check failed"
                            else:
                                reject_reason = "Person not registered"
                        else:
                            reject_reason = f"Low confidence (distance: {best_dist:.2f})"
                        
                        if is_verified:
                            found_someone = True
                            recognized_name = identity
                            recognized_confidence = confidence
                            
                            # Check if already marked today
                            already_marked = is_attendance_marked_today(identity)
                            
                            if already_marked:
                                # Yellow box - already marked
                                cv2.rectangle(display_img, (fx, fy), (fx+fw, fy+fh), (0, 255, 255), 3)
                                label = f"{identity} | {confidence:.0f}% | Marked"
                                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                cv2.rectangle(display_img, (fx, fy-th-15), (fx+tw+10, fy), (0, 255, 255), -1)
                                cv2.putText(display_img, label, (fx+5, fy-8), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                            else:
                                # Green box - new attendance
                                cv2.rectangle(display_img, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 3)
                                label = f"{identity} | {confidence:.0f}%"
                                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                cv2.rectangle(display_img, (fx, fy-th-15), (fx+tw+10, fy), (0, 255, 0), -1)
                                cv2.putText(display_img, label, (fx+5, fy-8), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                # Mark attendance
                                log_attendance(identity, confidence)
                        
                        if not is_verified:
                            # Show rejection reason in debug
                            if reject_reason:
                                st.info(f"ℹ️ Not recognized: {reject_reason}")
                            
                            # Check why verification failed
                            if "Liveness" in reject_reason:
                                # Face matched but LIVENESS FAILED - likely photo attack
                                spoof_detected = True
                                cv2.rectangle(display_img, (fx, fy), (fx+fw, fy+fh), (0, 165, 255), 3)  # Orange
                                label = f"SPOOF?"
                                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                cv2.rectangle(display_img, (fx, fy-th-15), (fx+tw+10, fy), (0, 165, 255), -1)
                                cv2.putText(display_img, label, (fx+5, fy-8), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            else:
                                # Unknown/unregistered face - Red box
                                cv2.rectangle(display_img, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 3)
                                label = f"NOT RECOGNIZED"
                                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                cv2.rectangle(display_img, (fx, fy-th-15), (fx+tw+10, fy), (0, 0, 255), -1)
                                cv2.putText(display_img, label, (fx+5, fy-8), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show result image
                st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Show recognition result below
                if found_someone:
                    if already_marked:
                        st.markdown(f"""
                        <div style="background: linear-gradient(45deg, #f39c12, #f1c40f); padding: 25px; border-radius: 15px; text-align: center; margin: 10px 0;">
                            <h2 style="color: #333; margin: 0;">👤 {recognized_name}</h2>
                            <h3 style="color: #333; margin: 10px 0;">🎯 Score: {recognized_confidence:.1f}%</h3>
                            <p style="color: #333; margin: 0;">⚠️ Already marked attendance today!</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background: linear-gradient(45deg, #11998e, #38ef7d); padding: 25px; border-radius: 15px; text-align: center; margin: 10px 0;">
                            <h2 style="color: white; margin: 0;">👤 {recognized_name}</h2>
                            <h3 style="color: white; margin: 10px 0;">🎯 Score: {recognized_confidence:.1f}%</h3>
                            <p style="color: white; margin: 0;">✅ Attendance Marked!</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    # Show why not recognized
                    if spoof_detected:
                        st.markdown(f"""
                        <div style="background: linear-gradient(45deg, #ff6b35, #f7931e); padding: 25px; border-radius: 15px; text-align: center; margin: 10px 0;">
                            <h2 style="color: white; margin: 0;">🚫 Possible Spoof</h2>
                            <p style="color: white; margin: 0;">Liveness check failed. Try with better lighting.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background: linear-gradient(45deg, #eb3349, #f45c43); padding: 25px; border-radius: 15px; text-align: center; margin: 10px 0;">
                            <h2 style="color: white; margin: 0;">❓ Face Not Recognized</h2>
                            <p style="color: white; margin: 0;">Not registered or no face detected</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
            except ImportError:
                st.error("❌ DeepFace not installed! Run: `python -m pip install deepface tf-keras`")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

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

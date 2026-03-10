# 🛡️ VisionAI Pro - Face Recognition & Attendance System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=for-the-badge&logo=streamlit)
![DeepFace](https://img.shields.io/badge/DeepFace-AI-purple?style=for-the-badge)

**A Professional Face Recognition System with Automatic Attendance Tracking**

[Features](#-features) • [Installation](#%EF%B8%8F-installation) • [Usage](#-how-to-use) • [Technical Details](#-technical-details) • [API Reference](#-api-reference)

</div>

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Features](#-features)
3. [System Architecture](#-system-architecture)
4. [Technical Details](#-technical-details)
   - [Face Detection Model](#1-face-detection-model-haar-cascade)
   - [Face Recognition Model](#2-face-recognition-model-arcface)
5. [Installation](#%EF%B8%8F-installation)
6. [Usage Guide](#-how-to-use)
7. [Project Structure](#-project-structure)
8. [API Reference](#-api-reference)
9. [Performance Optimization](#-performance-optimization)
10. [Troubleshooting](#-troubleshooting)

---

## 🎯 Overview

**VisionAI Pro** is a production-ready face recognition system designed for:
- **Organizations** - Employee attendance tracking
- **Educational Institutions** - Student attendance management
- **Security Systems** - Access control and verification

The system uses state-of-the-art deep learning models to achieve high accuracy while maintaining real-time performance.

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 📸 **Face Enrollment** | Register faces using webcam with live preview |
| 🔍 **Face Recognition** | Identify registered individuals with confidence scores |
| 📋 **Auto Attendance** | Automatic attendance logging upon recognition |
| 📊 **Analytics Dashboard** | View statistics and export reports |
| 🎨 **Modern UI** | Beautiful dark theme with gradient design |
| 📥 **CSV Export** | Download attendance records |
| 🔄 **Database Management** | Easy refresh and management of face database |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     VISIONAI PRO SYSTEM                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Webcam     │───▶│  Face        │───▶│  Face        │  │
│  │   Input      │    │  Detection   │    │  Recognition │  │
│  └──────────────┘    │  (Haar/CV)   │    │  (ArcFace)   │  │
│                      └──────────────┘    └──────┬───────┘  │
│                                                  │          │
│                                                  ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Streamlit  │◀───│  Attendance  │◀───│  Identity    │  │
│  │   Dashboard  │    │  Logger      │    │  Matching    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow:
1. **Input** → Webcam captures face image
2. **Detection** → Haar Cascade locates face region
3. **Embedding** → ArcFace extracts 512-D face embedding
4. **Matching** → Cosine similarity with database embeddings
5. **Output** → Identity + Confidence score + Attendance log

---

## 🔬 Technical Details

### 1. Face Detection Model (Haar Cascade)

| Property | Value |
|----------|-------|
| **Model Name** | Haar Cascade Classifier |
| **Type** | Machine Learning (Viola-Jones Algorithm) |
| **Input** | Grayscale Image |
| **Output** | Bounding Box Coordinates (x, y, w, h) |
| **Speed** | ~30 FPS (Real-time) |
| **File** | `haarcascade_frontalface_default.xml` |

#### How It Works:
```
Image → Grayscale Conversion → Integral Image → Haar Features → Cascade Classifier → Face Region
```

**Haar Features:**
- Edge features (vertical, horizontal)
- Line features
- Four-rectangle features

**Advantages:**
- ✅ Very fast detection
- ✅ Low computational cost
- ✅ No GPU required
- ✅ Built into OpenCV

**Limitations:**
- ❌ Frontal faces only
- ❌ Sensitive to lighting
- ❌ May miss tilted faces

---

### 2. Face Recognition Model (ArcFace)

| Property | Value |
|----------|-------|
| **Model Name** | ArcFace (Additive Angular Margin Loss) |
| **Architecture** | ResNet-100 Backbone |
| **Embedding Size** | 512 dimensions |
| **Training Data** | MS1MV2 (5.8M images, 85K identities) |
| **Accuracy** | 99.83% (LFW Benchmark) |
| **Framework** | DeepFace (TensorFlow/Keras) |

#### How It Works:
```
Face Image → CNN Feature Extraction → 512-D Embedding → Cosine Similarity → Identity Match
```

**Architecture Details:**
```
Input (112x112x3)
    │
    ▼
┌─────────────────┐
│  ResNet-100     │
│  Backbone       │
│  (Deep CNN)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Global Average │
│  Pooling        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Fully Connected│
│  Layer (512-D)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ArcFace Loss   │
│  (Training)     │
└────────┬────────┘
         │
         ▼
   512-D Face Embedding
```

**ArcFace Loss Function:**
```
L = -log(e^(s·cos(θ_yi + m)) / (e^(s·cos(θ_yi + m)) + Σ e^(s·cos(θ_j))))

Where:
- θ = angle between feature and weight
- m = angular margin (0.5 radians)
- s = feature scale (64)
```

**Why ArcFace?**
| Model | LFW Accuracy | Speed |
|-------|--------------|-------|
| FaceNet | 99.63% | Medium |
| VGG-Face | 98.95% | Slow |
| **ArcFace** | **99.83%** | **Fast** |
| DeepFace | 97.35% | Medium |

---

### 3. Similarity Matching

**Distance Metric:** Cosine Similarity

```python
similarity = 1 - (distance / threshold)
confidence = similarity × 100%
```

| Distance | Result |
|----------|--------|
| < 0.68 | ✅ Match (Verified) |
| ≥ 0.68 | ❌ No Match (Unknown) |

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- Webcam
- 4GB+ RAM recommended

### Step 1: Clone/Download Project
```bash
cd your-project-folder
```

### Step 2: Install Dependencies
```bash
python -m pip install deepface tf-keras opencv-python numpy streamlit pandas
```

### Step 3: Verify Installation
```bash
python -c "import deepface; import cv2; import streamlit; print('✅ All packages installed!')"
```

### Step 4: Run Application
```bash
python -m streamlit run advanced_web_app.py
```

---

## 📖 How to Use

### 1️⃣ Face Enrollment

1. Navigate to **📸 Enrollment** page
2. Enter person's name
3. Click camera button to capture photo
4. Verify face is detected (green box)
5. Click **"💾 Save Face"**

**Best Practices:**
- ☀️ Ensure good lighting
- 👤 Face the camera directly
- 📐 Take 2-3 photos from different angles

### 2️⃣ Face Recognition

1. Navigate to **🔍 Recognition** page
2. Click camera button
3. System will:
   - Detect face
   - Match against database
   - Display identity + confidence
   - Auto-mark attendance

### 3️⃣ View Attendance

1. Navigate to **📊 Attendance Log**
2. Filter by date (optional)
3. Export to CSV if needed

---

## 📁 Project Structure

```
📦 VisionAI-Pro/
├── 📄 advanced_web_app.py    # Main application (Streamlit)
├── 📄 README.md              # Documentation
├── 📄 requirements.txt       # Dependencies
├── 📂 faces/                 # Face database
│   ├── Ahmed_091523.jpg
│   ├── Sara_092045.jpg
│   └── ...
└── 📄 attendance_log.csv     # Auto-generated attendance records
```

---

## 🔌 API Reference

### Core Functions

| Function | Description | Parameters |
|----------|-------------|------------|
| `detect_face_haar(image)` | Detect faces in image | `image`: numpy array |
| `get_registered_faces()` | Get list of enrolled names | None |
| `refresh_database()` | Clear cached embeddings | None |
| `log_attendance(name, confidence)` | Record attendance | `name`: str, `confidence`: float |
| `get_attendance_data()` | Read attendance CSV | None |

### DeepFace Integration

```python
from deepface import DeepFace

# Face Recognition
results = DeepFace.find(
    img_path=image,           # Input image
    db_path="faces",          # Database folder
    model_name='ArcFace',     # Recognition model
    detector_backend='opencv', # Detection model
    enforce_detection=False,  # Don't raise error if no face
    silent=True               # Suppress logs
)
```

---

## ⚡ Performance Optimization

| Setting | Recommendation | Impact |
|---------|----------------|--------|
| **Detector** | OpenCV (Haar) | Fastest detection |
| **Model** | ArcFace | Best accuracy/speed ratio |
| **Image Size** | 640x480 | Optimal for webcam |
| **Lighting** | Well-lit room | Better detection |

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 8 GB |
| CPU | Dual Core | Quad Core |
| Storage | 500 MB | 1 GB |
| Camera | 720p | 1080p |

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No face detected" | Improve lighting, face camera directly |
| "Unknown" result | Re-enroll with better photos |
| Slow performance | Close other applications |
| Camera not working | Check permissions, restart browser |
| DeepFace error | Run: `pip install deepface tf-keras` |

---

## 📊 Attendance File Format

**File:** `attendance_log.csv`

| Column | Type | Description |
|--------|------|-------------|
| Name | String | Person's name |
| Date | YYYY-MM-DD | Date of attendance |
| Time | HH:MM:SS | Time of recognition |
| Confidence | Percentage | Match confidence |

**Sample:**
```csv
Name,Date,Time,Confidence
Ahmed Khan,2026-03-10,09:15:23,87.5%
Sara Ali,2026-03-10,09:20:45,92.1%
```

---

## 📚 References

1. **ArcFace Paper:** Deng, J., et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." CVPR 2019.
2. **DeepFace Library:** https://github.com/serengil/deepface
3. **OpenCV Documentation:** https://docs.opencv.org/
4. **Streamlit Documentation:** https://docs.streamlit.io/

---

## 📄 License

This project is for educational purposes.

---

<div align="center">

**Built with ❤️ using DeepFace, OpenCV & Streamlit**

⭐ Star this project if you found it helpful!

</div>

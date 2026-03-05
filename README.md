# 🛡️ Professional Real-Time Face Recognition System

Yeh aik advanced Computer Vision project hay jo Real-Time chehron ko detect karta hay aur unki pehchan (Identification) karta hay. Is system ko aise design kiya gaya hay ke yeh **Masks** aur **Side-Profiles** par bhi behtreen kaam kary.

## 🚀 Key Features
- **Face Detection:** MTCNN (Landmark-based detection).
- **Face Recognition:** ArcFace (InsightFace's high-precision model).
- **Real-Time FPS:** Optimized for Windows using DirectShow (CAP_DSHOW).
- **Robustness:** Handles partially covered faces (Masks) and side views.
- **Smart Database:** Auto-refreshes when new faces are added.
- **Accuracy Display:** Shows confidence percentage for every match.

## 🛠️ Tech Stack
- **Python 3.10+**
- **DeepFace Framework** (Backend for Recognition)
- **OpenCV** (Image processing & Camera handling)
- **ArcFace & MTCNN** (Deep Learning Models)

## 📁 Project Structure
- `app.py`: Main real-time recognition script.
- `capture.py`: Script to add new people to the system.
- `faces/`: Folder containing reference images of known people.
- `requirements.txt`: List of all dependencies.

## ⚙️ Installation & Usage

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add Your Face:**
   Run the capture script and enter your name. Press 's' to save your photo.
   ```bash
   python capture.py
   ```
   *Tip: 3-4 photos save karein mukhtalif angles (Front, Side, Mask) ke saath behtar results ke liye.*

3. **Run Recognition:**
   ```bash
   python app.py
   ```
   Press **'q'** or click the **'X'** on window to quit.

## 🔒 Accuracy Tips
- Light achi honi chahiye.
- `faces/` folder mein images clear honi chahiye.
- Agar system "Unknown" kahay, to `app.py` mein `distance < 0.68` ki value ko thora barha den (e.g. 0.70).

---
**Developed with ❤️ by Your Computer Vision Expert AI.**

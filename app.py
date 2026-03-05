import cv2
from deepface import DeepFace
import os

# Database folder
DB_PATH = "faces"
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

print("--- Face Recognition Started ---")
print("Press '1': RetinaFace (Precise), '2': MTCNN (Standard), 'q': Quit")
detector_backend = 'mtcnn'

# Database Refresh - Delete all .pkl files to ensure fresh start
for f in os.listdir(DB_PATH):
    if f.endswith(".pkl"):
        os.remove(os.path.join(DB_PATH, f))
        print(f"Refreshed: {f}")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Improved find call with explicit alignment
        results = DeepFace.find(img_path=frame, 
                                db_path=DB_PATH, 
                                model_name='ArcFace', 
                                detector_backend=detector_backend, 
                                enforce_detection=False,
                                align=True,
                                silent=True)
        
        if len(results) > 0 and not results[0].empty:
            match = results[0].iloc[0]
            dist = match['distance']
            confidence = max(0, min(100, (1 - dist/0.8) * 100))
            x, y, w, h = int(match['source_x']), int(match['source_y']), int(match['source_w']), int(match['source_h'])
            
            if dist < 0.68: # Strength of match
                name = os.path.basename(match['identity']).split('.')[0]
                name = ''.join([i for i in name if not i.isdigit()]).capitalize()
                color, label = (0, 255, 0), f"{name} {confidence:.1f}%"
            else:
                color, label = (0, 0, 255), "Unknown"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y - 30), (x + w, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    except Exception as e:
        pass

    cv2.putText(frame, f"Detector: {detector_backend} (1: Retina, 2: MTCNN)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imshow("DeepFace Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty("DeepFace Recognition", cv2.WND_PROP_VISIBLE) < 1:
        break
    elif key == ord('1'):
        detector_backend = 'retinaface'
        print("Switched to RetinaFace")
    elif key == ord('2'):
        detector_backend = 'mtcnn'
        print("Switched to MTCNN")

cap.release()
cv2.destroyAllWindows()

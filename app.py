import cv2
from deepface import DeepFace
import os

# Database folder
DB_PATH = "faces"
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

print("--- Face Recognition Started ---")
print("Press 'q' to quit.")

# 1. Database Refresh Logic
# DeepFace creates a .pkl file in the faces folder. Agar naye chehre add kiye hain, 
# to purani file delete karni hogi taake database refresh ho jaye.
for f in os.listdir(DB_PATH):
    if f.endswith(".pkl"):
        os.remove(os.path.join(DB_PATH, f))
        print(f"Database refresh kiya gaya: {f} deleted")

# Camera stream
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("--- Face Recognition Started (YOLOv8 + ArcFace) ---")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera nahi mil raha!")
        break

    try:
        # Recognition call
        results = DeepFace.find(img_path=frame, 
                                db_path=DB_PATH, 
                                model_name='ArcFace', 
                                detector_backend='mtcnn', 
                                enforce_detection=False,
                                silent=True)
        
        if len(results) > 0 and not results[0].empty:
            match = results[0].iloc[0]
            dist = match['distance']
            
            # Distance ko Percentage mein convert karna (Scale: 0.0 to 1.0)
            # ArcFace mein 0.68 se niche match hota hay.
            confidence = max(0, min(100, (1 - dist/0.8) * 100))
            
            x, y, w, h = int(match['source_x']), int(match['source_y']), int(match['source_w']), int(match['source_h'])
            
            if dist < 0.68: # Threshold check
                name = os.path.basename(match['identity']).split('.')[0]
                name = ''.join([i for i in name if not i.isdigit()]).capitalize()
                
                color = (0, 255, 0) # Green for known
                label = f"{name} {confidence:.1f}%"
            else:
                color = (0, 0, 255) # Red for unknown
                label = "Unknown"

            # Draw Beautiful Box & Label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y - 30), (x + w, y), color, -1) # Label Background
            cv2.putText(frame, label, (x + 5, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    except Exception as e:
        pass

    # Screen par Quit ka hint dena
    cv2.putText(frame, "Press 'q' to Quit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("DeepFace Recognition", frame)

    # 1. 'q' key se band karna
    # 2. Window ka 'X' button press karne se band karna
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty("DeepFace Recognition", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()

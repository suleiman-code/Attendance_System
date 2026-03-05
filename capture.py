import cv2
import os

# Create faces directory if not exists
DB_PATH = "faces"
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

name = input("Apna naam enter karein: ").strip()
if not name:
    print("Naam lazmi hay!")
    exit()

print("Camera start ho raha hay...")
# CAP_DSHOW selection windows ke liye zyada stable hay
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Camera open nahi ho saka! Check karen koi aur app to use nahi kar rahi.")
    exit()

print(f"\n{name}, camera ki taraf dekhen.")
print("Photo save karny ke liye keyboard par 's' press karein.")
print("Band karne ke liye 'q' press karein.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow("Capture Face", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        img_name = os.path.join(DB_PATH, f"{name}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Photo saved as {img_name}")
        break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

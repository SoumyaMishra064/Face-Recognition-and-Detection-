import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if not os.path.exists("dataset"):
    os.makedirs("dataset")

cap = cv2.VideoCapture(0)
user_id = input("Enter User ID (numeric): ")
img_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Register Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('c'): 
        img_count += 1
        img_path = f"dataset/user_{user_id}_{img_count}.jpg"
        cv2.imwrite(img_path, face)
        print(f"Image {img_count} saved!")

    if img_count >= 10: 
        print("Face registration complete!")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()

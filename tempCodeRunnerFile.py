import cv2
import numpy as np
import os


try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    print("Error: cv2.face module not found. Install opencv-contrib-python")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

dataset_path = "dataset"
images, labels = [], []

for file in os.listdir(dataset_path):
    if file.endswith(".jpg"):
        img_path = os.path.join(dataset_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:  # Ensure image is loaded
            img = cv2.resize(img, (200, 200))  # Resize to a fixed size
            user_id = int(file.split("_")[1])  
            images.append(img)
            labels.append(user_id)
        else:
            print(f"Error loading image: {img_path}")

if len(images) == 0:
    print("No images found in dataset!")
    exit()

images, labels = np.array(images), np.array(labels)

recognizer.train(images, labels)
recognizer.save("face_trainer.yml")
print("Training completed and model saved!")

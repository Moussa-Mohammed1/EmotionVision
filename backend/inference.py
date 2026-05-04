import cv2
import torch
from PIL import Image
from utils.preprocessing import get_transform


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


model = torch.load("models/emotion_model.pth")
model.eval()

transform = get_transform()

emotion_labels = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        face_pil = Image.fromarray(face)

        face_tensor = transform(face_pil)
        face_tensor = face_tensor.unsqueeze(0)  

        with torch.no_grad():
            output = model(face_tensor)
            _, predicted = torch.max(output, 1)
            emotion = emotion_labels[predicted.item()]

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255,0,0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
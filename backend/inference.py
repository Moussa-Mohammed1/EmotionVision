from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from utils.preprocessing import get_transform


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "emotion_model.pth"
CASCADE_PATH = BASE_DIR / "models" / "haarcascade_frontalface_default.xml"

EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
transform = get_transform()
_model = None
_model_load_error: Exception | None = None


def get_model() -> torch.nn.Module:
    global _model, _model_load_error

    if _model is not None:
        return _model

    if _model_load_error is not None:
        raise RuntimeError(
            f"Emotion model is unavailable: {_model_load_error}"
        ) from _model_load_error

    if not MODEL_PATH.exists():
        _model_load_error = FileNotFoundError(f"Missing model file: {MODEL_PATH}")
        raise RuntimeError(
            f"Emotion model is unavailable: {_model_load_error}"
        ) from _model_load_error

    if MODEL_PATH.stat().st_size == 0:
        _model_load_error = ValueError(f"Model file is empty: {MODEL_PATH}")
        raise RuntimeError(
            f"Emotion model is unavailable: {_model_load_error}"
        ) from _model_load_error

    try:
        _model = torch.load(MODEL_PATH, map_location="cpu")
        _model.eval()
        return _model
    except Exception as error:
        _model_load_error = error
        raise RuntimeError(f"Emotion model is unavailable: {error}") from error


def predict_emotion(face_image: np.ndarray) -> tuple[str, float]:
    if face_image is None:
        raise ValueError("face_image cannot be empty")

    model = get_model()
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_tensor = transform(face_rgb).unsqueeze(0)

    with torch.no_grad():
        output = model(face_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    label = EMOTION_LABELS[predicted.item()]
    return label, float(confidence.item())


def detect_faces(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def analyze_frame(frame: np.ndarray) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []

    for x, y, w, h in detect_faces(frame):
        face = frame[y : y + h, x : x + w]
        emotion, confidence = predict_emotion(face)
        results.append(
            {
                "bbox": [x, y, w, h],
                "emotion": emotion,
                "confidence": confidence,
            }
        )

    return results
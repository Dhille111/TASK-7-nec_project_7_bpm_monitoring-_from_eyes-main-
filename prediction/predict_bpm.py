import cv2
import dlib
import numpy as np

from tensorflow.keras.models import load_model

# ==========================
# Load Trained CNN
# ==========================

model = load_model(
    "saved_models/cnn_model.keras"
)

# ==========================
# Face Detector
# ==========================

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(
    "models/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat"
)

LEFT_EYE = list(range(36, 42))

# ==========================
# Signal Buffer
# ==========================

signal_buffer = []

BUFFER_SIZE = 300

# ==========================
# Webcam
# ==========================

cap = cv2.VideoCapture(0)

print("Starting BPM Prediction...")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = detector(gray)

    for face in faces:

        landmarks = predictor(
            gray,
            face
        )

        eye_pts = []

        for idx in LEFT_EYE:

            x = landmarks.part(idx).x
            y = landmarks.part(idx).y

            eye_pts.append((x, y))

        eye_pts = np.array(eye_pts)

        x, y, w, h = cv2.boundingRect(
            eye_pts
        )

        pad = 15

        eye = frame[
            max(0, y-pad):y+h+pad,
            max(0, x-pad):x+w+pad
        ]

        if eye.size == 0:
            continue

        eye = cv2.resize(
            eye,
            (128, 128)
        )

        green_signal = np.mean(
            eye[:, :, 1]
        )

        signal_buffer.append(
            green_signal
        )

        if len(signal_buffer) > BUFFER_SIZE:
            signal_buffer.pop(0)

        cv2.imshow(
            "Eye ROI",
            eye
        )

    predicted_bpm = 0

    if len(signal_buffer) == BUFFER_SIZE:

        signal = np.array(
            signal_buffer
        )

        signal = signal.reshape(
            1,
            300,
            1
        )

        predicted_bpm = model.predict(
            signal,
            verbose=0
        )[0][0]

        cv2.putText(
            frame,
            f"Predicted BPM: {int(predicted_bpm)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    cv2.putText(
        frame,
        f"Samples: {len(signal_buffer)}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )

    cv2.imshow(
        "Retina Heart Rate Prediction",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
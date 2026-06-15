import cv2
import dlib
import numpy as np
import time

# ==================================
# Face Detector & Landmark Predictor
# ==================================
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(
    "models/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat"
)

LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# ==================================
# Webcam
# ==================================
cap = cv2.VideoCapture(0)

# Better FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:

        landmarks = predictor(gray, face)

        # ==================================
        # LEFT EYE
        # ==================================
        left_points = []

        for idx in LEFT_EYE:

            x = landmarks.part(idx).x
            y = landmarks.part(idx).y

            left_points.append((x, y))

        left_points = np.array(left_points)

        lx, ly, lw, lh = cv2.boundingRect(left_points)

        pad_x = 15
        pad_y = 15

        x1 = max(0, lx - pad_x)
        y1 = max(0, ly - pad_y)

        x2 = min(frame.shape[1], lx + lw + pad_x)
        y2 = min(frame.shape[0], ly + lh + pad_y)

        left_eye = frame[y1:y2, x1:x2]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        left_center_x = lx + lw // 2
        left_center_y = ly + lh // 2

        # ==================================
        # RIGHT EYE
        # ==================================
        right_points = []

        for idx in RIGHT_EYE:

            x = landmarks.part(idx).x
            y = landmarks.part(idx).y

            right_points.append((x, y))

        right_points = np.array(right_points)

        rx, ry, rw, rh = cv2.boundingRect(right_points)

        rx1 = max(0, rx - pad_x)
        ry1 = max(0, ry - pad_y)

        rx2 = min(frame.shape[1], rx + rw + pad_x)
        ry2 = min(frame.shape[0], ry + rh + pad_y)

        right_eye = frame[ry1:ry2, rx1:rx2]

        cv2.rectangle(
            frame,
            (rx1, ry1),
            (rx2, ry2),
            (255, 0, 0),
            2
        )

        right_center_x = rx + rw // 2
        right_center_y = ry + rh // 2

        # ==================================
        # Eye Centers
        # ==================================
        cv2.circle(
            frame,
            (left_center_x, left_center_y),
            3,
            (0, 255, 255),
            -1
        )

        cv2.circle(
            frame,
            (right_center_x, right_center_y),
            3,
            (0, 255, 255),
            -1
        )

        # ==================================
        # Display Left Eye
        # ==================================
        if left_eye.size > 0:

            left_eye = cv2.resize(
                left_eye,
                (128, 128)
            )

            cv2.imshow(
                "Left Eye",
                left_eye
            )

        # ==================================
        # Display Right Eye
        # ==================================
        if right_eye.size > 0:

            right_eye = cv2.resize(
                right_eye,
                (128, 128)
            )

            cv2.imshow(
                "Right Eye",
                right_eye
            )

        # ==================================
        # Combined ROI
        # ==================================
        if left_eye.size > 0 and right_eye.size > 0:

            combined_eye = np.hstack(
                [left_eye, right_eye]
            )

            cv2.imshow(
                "Combined Eye ROI",
                combined_eye
            )

    # ==================================
    # FPS
    # ==================================
    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Frame",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
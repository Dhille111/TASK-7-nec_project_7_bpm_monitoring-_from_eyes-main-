import cv2
import dlib
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt
import time

# ==================================
# Face Detector
# ==================================
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(
    "models/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat"
)

LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# ==================================
# Signal Buffers
# ==================================
signal_buffer = deque(maxlen=300)
bpm_history = deque(maxlen=20)

# ==================================
# Bandpass Filter
# 0.8Hz - 3Hz
# 48 BPM - 180 BPM
# ==================================
def bandpass_filter(signal, fs):

    low = 0.8
    high = 3.0

    nyquist = 0.5 * fs

    low /= nyquist
    high /= nyquist

    b, a = butter(
        3,
        [low, high],
        btype="band"
    )

    return filtfilt(
        b,
        a,
        signal
    )

# ==================================
# Webcam
# ==================================
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = time.time()

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

    bpm_display = "--"

    for face in faces:

        landmarks = predictor(gray, face)

        # ==================================
        # LEFT EYE ROI
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

        # ==================================
        # RIGHT EYE ROI
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

        right_eye = frame[
            ry1:ry2,
            rx1:rx2
        ]

        if left_eye.size > 0 and right_eye.size > 0:

            left_eye = cv2.resize(
                left_eye,
                (128, 128)
            )

            right_eye = cv2.resize(
                right_eye,
                (128, 128)
            )

            combined_eye = np.hstack(
                [left_eye, right_eye]
            )

            # ==================================
            # Green Channel Signal
            # ==================================
            green_signal = np.mean(
                combined_eye[:, :, 1]
            )

            signal_buffer.append(
                green_signal
            )

            cv2.imshow(
                "Combined Eye ROI",
                combined_eye
            )

            # ==================================
            # Heart Rate Estimation
            # ==================================
            if len(signal_buffer) > 150:

                signal = np.array(
                    signal_buffer
                )

                # Remove DC component
                signal = signal - np.mean(signal)

                # Normalize
                signal = signal / (
                    np.std(signal) + 1e-8
                )

                fps_estimate = max(15, int(fps))

                filtered = bandpass_filter(
                    signal,
                    fps_estimate
                )

                fft = np.abs(
                    np.fft.rfft(filtered)
                )

                freqs = np.fft.rfftfreq(
                    len(filtered),
                    d=1/fps_estimate
                )

                # Physiological HR Range
                mask = (
                    (freqs >= 0.8) &
                    (freqs <= 2.0)
                )

                if np.any(mask):

                    peak_freq = freqs[mask][
                        np.argmax(
                            fft[mask]
                        )
                    ]

                    bpm = peak_freq * 60

                    # Reject impossible BPM
                    if 50 <= bpm <= 110:

                        bpm_history.append(
                            bpm
                        )

                        bpm_display = int(
                            np.mean(
                                bpm_history
                            )
                        )

    # ==================================
    # FPS
    # ==================================
    current_time = time.time()

    fps = 1 / (
        current_time - prev_time
    )

    prev_time = current_time

    # ==================================
    # Display
    # ==================================
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )

    cv2.putText(
        frame,
        f"BPM: {bpm_display}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow(
        "Heart Rate Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
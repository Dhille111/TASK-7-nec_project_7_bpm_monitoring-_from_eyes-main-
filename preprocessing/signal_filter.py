import cv2
import dlib
import numpy as np
import time

from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.fft import fft

from collections import deque

# ==========================================
# Face Detector
# ==========================================

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(
    "models/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat"
)

LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# ==========================================
# Signal Buffers
# ==========================================

signal_buffer = []

BUFFER_SIZE = 300

bpm_history = deque(maxlen=15)

fps_measurements = []

# ==========================================
# Bandpass Filter
# ==========================================

def butter_bandpass(signal, fs):

    lowcut = 0.8
    highcut = 2.0

    nyquist = fs * 0.5

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        4,
        [low, high],
        btype="band"
    )

    filtered = filtfilt(
        b,
        a,
        signal
    )

    return filtered


# ==========================================
# BPM Estimation
# ==========================================

def estimate_bpm(signal, fs):

    signal = signal - np.mean(signal)

    spectrum = np.abs(
        fft(signal)
    )

    freqs = np.fft.fftfreq(
        len(signal),
        d=1/fs
    )

    positive = freqs > 0

    freqs = freqs[positive]
    spectrum = spectrum[positive]

    valid = (
        (freqs >= 0.8) &
        (freqs <= 2.0)
    )

    freqs = freqs[valid]
    spectrum = spectrum[valid]

    if len(freqs) == 0:
        return 0

    peak_idx = np.argmax(
        spectrum
    )

    peak_freq = freqs[
        peak_idx
    ]

    bpm = peak_freq * 60

    return bpm


# ==========================================
# Webcam
# ==========================================

cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

prev_time = time.time()

# ==========================================
# Main Loop
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(
        frame,
        1
    )

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = detector(
        gray
    )

    bpm = 0

    for face in faces:

        landmarks = predictor(
            gray,
            face
        )

        # ==================================
        # LEFT EYE ROI
        # ==================================

        left_points = []

        for idx in LEFT_EYE:

            x = landmarks.part(idx).x
            y = landmarks.part(idx).y

            left_points.append(
                (x, y)
            )

        left_points = np.array(
            left_points
        )

        lx, ly, lw, lh = cv2.boundingRect(
            left_points
        )

        pad_x = 15
        pad_y = 15

        x1 = max(
            0,
            lx - pad_x
        )

        y1 = max(
            0,
            ly - pad_y
        )

        x2 = min(
            frame.shape[1],
            lx + lw + pad_x
        )

        y2 = min(
            frame.shape[0],
            ly + lh + pad_y
        )

        left_eye = frame[
            y1:y2,
            x1:x2
        ]

        # ==================================
        # RIGHT EYE ROI
        # ==================================

        right_points = []

        for idx in RIGHT_EYE:

            x = landmarks.part(idx).x
            y = landmarks.part(idx).y

            right_points.append(
                (x, y)
            )

        right_points = np.array(
            right_points
        )

        rx, ry, rw, rh = cv2.boundingRect(
            right_points
        )

        rx1 = max(
            0,
            rx - pad_x
        )

        ry1 = max(
            0,
            ry - pad_y
        )

        rx2 = min(
            frame.shape[1],
            rx + rw + pad_x
        )

        ry2 = min(
            frame.shape[0],
            ry + rh + pad_y
        )

        right_eye = frame[
            ry1:ry2,
            rx1:rx2
        ]

        if left_eye.size == 0 or right_eye.size == 0:
            continue

        left_eye = cv2.resize(
            left_eye,
            (128, 128)
        )

        right_eye = cv2.resize(
            right_eye,
            (128, 128)
        )

        combined_eye = np.hstack(
            [
                left_eye,
                right_eye
            ]
        )

        cv2.imshow(
            "Combined Eye ROI",
            combined_eye
        )

        # ==================================
        # Motion Rejection
        # ==================================

        motion_score = np.std(
            combined_eye[:, :, 1]
        )

        if motion_score > 70:
            continue

        # ==================================
        # Green Channel Signal
        # ==================================

        green_signal = np.mean(
            combined_eye[:, :, 1].astype(
                np.float32
            )
        )

        signal_buffer.append(
            green_signal
        )

        if len(signal_buffer) > BUFFER_SIZE:
            signal_buffer.pop(0)

        # ==================================
        # BPM Estimation
        # ==================================

        if len(signal_buffer) == BUFFER_SIZE:

            if len(fps_measurements) >= 30:

                fs = np.mean(
                    fps_measurements[-30:]
                )

            else:

                fs = 20

            signal = np.array(
                signal_buffer,
                dtype=np.float32
            )

            signal = (
                signal - np.mean(signal)
            ) / (
                np.std(signal) + 1e-8
            )

            filtered_signal = butter_bandpass(
                signal,
                fs
            )

            bpm_estimate = estimate_bpm(
                filtered_signal,
                fs
            )

            if 50 <= bpm_estimate <= 120:

                bpm_history.append(
                    bpm_estimate
                )

            if len(bpm_history) > 0:

                bpm = np.mean(
                    bpm_history
                )

    # ==================================
    # FPS
    # ==================================

    current_time = time.time()

    fps = 1 / (
        current_time - prev_time
    )

    prev_time = current_time

    fps_measurements.append(
        fps
    )

    # ==================================
    # Display
    # ==================================

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    if bpm > 0:

        cv2.putText(
            frame,
            f"BPM: {int(round(bpm))}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Heart Rate Detection",
        frame
    )

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
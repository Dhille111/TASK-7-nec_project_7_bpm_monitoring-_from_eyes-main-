import cv2
import dlib
import numpy as np
import time

from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.fft import fft

from dataset_builder import save_sample

# ==========================================
# Face Detector
# ==========================================

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(
    "models/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat"
)

LEFT_EYE = list(range(36, 42))

# ==========================================
# Signal Buffer
# ==========================================

signal_buffer = []

BUFFER_SIZE = 300

fps_measurements = []

# ==========================================
# Bandpass Filter
# ==========================================

def butter_bandpass(signal, fs):

    lowcut = 0.7
    highcut = 4.0

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        4,
        [low, high],
        btype="band"
    )

    return filtfilt(b, a, signal)

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

    peak_idx = np.argmax(
        spectrum
    )

    peak_freq = freqs[peak_idx]

    bpm = peak_freq * 60

    return bpm

# ==========================================
# Webcam
# ==========================================

cap = cv2.VideoCapture(0)

prev_time = time.time()

print("\nPress 's' to Auto Save Dataset")
print("Press ESC to Exit\n")

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

            eye_pts.append(
                (x, y)
            )

        eye_pts = np.array(
            eye_pts
        )

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

    # FPS

    current_time = time.time()

    fps = 1 / (
        current_time - prev_time
    )

    prev_time = current_time

    fps_measurements.append(
        fps
    )

    if len(fps_measurements) > 100:
        fps_measurements.pop(0)

    # BPM

    bpm = 0

    if len(signal_buffer) == BUFFER_SIZE:

        fs = np.mean(
            fps_measurements
        )

        filtered = butter_bandpass(
            np.array(signal_buffer),
            fs
        )

        bpm = estimate_bpm(
            filtered,
            fs
        )

        cv2.putText(
            frame,
            f"BPM: {int(bpm)}",
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
        "Auto Dataset Collector",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # ======================================
    # SAVE
    # ======================================

    if key == ord('s'):

        if len(signal_buffer) == BUFFER_SIZE:

            save_sample(
                np.array(signal_buffer),
                int(bpm)
            )

            print(
                f"Auto Saved BPM={int(bpm)}"
            )

        else:

            print(
                "Buffer Not Full Yet"
            )

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
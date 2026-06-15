import cv2
import dlib
import numpy as np

from dataset_builder import save_sample

# ==================================
# Dlib
# ==================================

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(
    "models/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat"
)

LEFT_EYE = list(range(36, 42))

# ==================================
# Dataset Buffer
# ==================================

frames = []

SEQUENCE_LENGTH = 300

# ==================================
# Webcam
# ==================================

cap = cv2.VideoCapture(0)

print("\nPress 's' to Save Sample")
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

        left_pts = []

        for idx in LEFT_EYE:

            x = landmarks.part(idx).x
            y = landmarks.part(idx).y

            left_pts.append((x, y))

        left_pts = np.array(left_pts)

        lx, ly, lw, lh = cv2.boundingRect(
            left_pts
        )

        pad = 15

        eye = frame[
            max(0, ly-pad):ly+lh+pad,
            max(0, lx-pad):lx+lw+pad
        ]

        if eye.size == 0:
            continue

        eye = cv2.resize(
            eye,
            (128, 128)
        )

        cv2.imshow(
            "Eye ROI",
            eye
        )

        green_signal = np.mean(
            eye[:, :, 1]
        )

        frames.append(
            green_signal
        )

        if len(frames) > SEQUENCE_LENGTH:
            frames.pop(0)

        # Debug Count
        cv2.putText(
            frame,
            f"Samples: {len(frames)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Dataset Collector",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # ==================================
    # SAVE SAMPLE
    # ==================================

    if key == ord('s'):

        print(
            f"\nCurrent Samples: {len(frames)}"
        )

        if len(frames) >= SEQUENCE_LENGTH:

            bpm = input(
                "\nEnter BPM Label: "
            )

            save_sample(
                np.array(frames),
                bpm
            )

            print(
                "\nSample Saved Successfully!"
            )

        else:

            print(
                f"\nNeed {SEQUENCE_LENGTH} samples."
            )

    # ==================================
    # EXIT
    # ==================================

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
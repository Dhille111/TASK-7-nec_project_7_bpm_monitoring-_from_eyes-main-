import cv2
import dlib

# Face detector
detector = dlib.get_frontal_face_detector()

# Landmark predictor
predictor = dlib.shape_predictor(
    "models/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat"
)


cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:

        landmarks = predictor(gray, face)

        # Draw all 68 points
        for n in range(68):

            x = landmarks.part(n).x
            y = landmarks.part(n).y

            cv2.circle(
                frame,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )

        # Highlight eyes
        for n in range(36, 48):

            x = landmarks.part(n).x
            y = landmarks.part(n).y

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 0, 255),
                -1
            )

    cv2.imshow("Eye Landmarks", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
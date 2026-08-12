import time
import cv2
import mediapipe as mp
import pygame
from ultralytics import YOLO

from blink_utils import LEFT_EYE, RIGHT_EYE, eye_aspect_ratio


# =====================================================
# ALARM SETUP
# =====================================================

pygame.mixer.init()


# =====================================================
# YOLO MODELS
# =====================================================

model = YOLO("yolov8n.pt")
alcohol_model = YOLO("runs/detect/alcohol_detector-6/weights/best.pt")

# =====================================================
# MEDIAPIPE FACE MESH
# =====================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# =====================================================
# CAMERA
# =====================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()


# =====================================================
# VARIABLES
# =====================================================

drowsy_start = None

blink_count = 0
blink_frames = 0

EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 2

# Mobile detection distance threshold
PHONE_DISTANCE_THRESHOLD = 250

# Emotion
emotion = "Unknown"


# =====================================================
# MAIN LOOP
# =====================================================

while True:
    # -------------------------------------------------
    # READ CAMERA FRAME
    # -------------------------------------------------

    success, frame = cap.read()

    if not success:
        print("Camera frame not received.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Alarm flags
    drowsy_alert = False
    mobile_detected = False
    alcohol_detected = False

    # =================================================
    # FACE MESH PROCESSING
    # =================================================

    results = face_mesh.process(rgb)

    # Default values
    face_detected = False
    face_x = None
    face_y = None

    ear = 0.0
    status = "Unknown"
    color = (255, 255, 255)

    # =================================================
    # FACE + DROWSINESS DETECTION
    # =================================================

    if results.multi_face_landmarks:
        face_detected = True
        face_landmarks = results.multi_face_landmarks[0].landmark

        # ---------------------------------------------
        # FACE CENTER
        # ---------------------------------------------

        face_x = int(face_landmarks[1].x * frame.shape[1])
        face_y = int(face_landmarks[1].y * frame.shape[0])

        # ---------------------------------------------
        # EAR CALCULATION
        # ---------------------------------------------

        leftEAR = eye_aspect_ratio(face_landmarks, LEFT_EYE)
        rightEAR = eye_aspect_ratio(face_landmarks, RIGHT_EYE)

        ear = (leftEAR + rightEAR) / 2

        # ---------------------------------------------
        # EYE STATUS
        # ---------------------------------------------

        if ear < EAR_THRESHOLD:
            blink_frames += 1
            status = "Eyes Closed"
            color = (0, 0, 255)
        else:
            if blink_frames >= CONSEC_FRAMES:
                blink_count += 1

            blink_frames = 0
            status = "Eyes Open"
            color = (0, 255, 0)

        # ---------------------------------------------
        # DROWSINESS DETECTION
        # ---------------------------------------------

        if ear < EAR_THRESHOLD:
            if drowsy_start is None:
                drowsy_start = time.time()

            elapsed = time.time() - drowsy_start

            # Eyes continuously closed for 3 seconds
            if elapsed >= 3:
                drowsy_alert = True
                cv2.putText(
                    frame,
                    "DROWSINESS ALERT!",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )
        else:
            # Eyes opened -> reset drowsiness
            drowsy_start = None

        # ---------------------------------------------
        # DISPLAY EAR, EYE STATUS & BLINKS
        # ---------------------------------------------

        cv2.putText(
            frame,
            f"EAR : {ear:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            status,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

        cv2.putText(
            frame,
            f"Blinks : {blink_count}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2,
        )

    else:
        # ---------------------------------------------
        # NO FACE DETECTED
        # ---------------------------------------------

        cv2.putText(
            frame,
            "FACE NOT DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        # Reset drowsiness
        drowsy_start = None

    # =================================================
    # YOLO OBJECT DETECTION (MOBILE PHONE)
    # =================================================

    yolo_results = model(frame, verbose=False)

    for result in yolo_results:
        for box in result.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]

            if name == "cell phone":
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                phone_x = (x1 + x2) // 2
                phone_y = (y1 + y2) // 2

                # Check proximity to face
                if face_detected:
                    distance = (
                        (face_x - phone_x) ** 2 + (face_y - phone_y) ** 2
                    ) ** 0.5

                    if distance < PHONE_DISTANCE_THRESHOLD:
                        mobile_detected = True
                else:
                    mobile_detected = True

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "MOBILE DETECTED",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

    # =================================================
    # MOBILE SAFETY WARNING
    # =================================================

    if mobile_detected:
        cv2.putText(
            frame,
            "DON'T USE MOBILE WHILE DRIVING",
            (20, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    # =================================================
    # ALCOHOL BOTTLE DETECTION
    # =================================================

    alcohol_results = alcohol_model(frame, verbose=False)

    for result in alcohol_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            if confidence > 0.50:
                alcohol_detected = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    "ALCOHOL BOTTLE",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

    # =================================================
    # ALCOHOL SAFETY WARNING
    # =================================================

    if alcohol_detected:
        cv2.putText(
            frame,
            "ALCOHOL DETECTED!",
            (20, frame.shape[0] - 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3,
        )

    # =================================================
    # AUDIO ALARM
    # =================================================

    if drowsy_alert or mobile_detected or alcohol_detected:
        if not pygame.mixer.music.get_busy():
            try:
                pygame.mixer.music.load("alarm.wav")
                pygame.mixer.music.play(-1)
            except Exception as e:
                print("Alarm Error:", e)
    else:
        pygame.mixer.music.stop()

    # =================================================
    # DISPLAY CAMERA WINDOW & EXIT
    # =================================================

    cv2.imshow("Driver Safety", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =====================================================
# CLEANUP
# =====================================================

pygame.mixer.music.stop()
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
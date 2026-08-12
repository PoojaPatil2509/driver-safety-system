import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
# =========================================================
# 1. INITIALIZATION
# =========================================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
# Pygame audio
pygame.mixer.init()
ALARM_FILE = "alarm.wav"
try:
    alarm = pygame.mixer.Sound(ALARM_FILE)
except Exception:
    alarm = None
    print("Warning: alarm.wav not found.")
# =========================================================
# 2. SETTINGS
# =========================================================
EAR_THRESHOLD = 0.25
DROWSINESS_TIME = 3.0
eye_closed_start = None
alarm_on = False
# =========================================================
# 3. EYE LANDMARKS
# =========================================================
# Left eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
# Right eye landmarks
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# =========================================================
# 4. EYE ASPECT RATIO (EAR)
# =========================================================
def calculate_ear(landmarks, eye_indices, width, height):
    points = []
    for index in eye_indices:
        x = int(landmarks[index].x * width)
        y = int(landmarks[index].y * height)
        points.append(np.array([x, y]))
    p1, p2, p3, p4, p5, p6 = points
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return 0
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear
# =========================================================
# 5. WEBCAM
# =========================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
# =========================================================
# 6. MAIN LOOP
# =========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read webcam frame.")
        break
    # Mirror image
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    # BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)
    # =====================================================
    # 7. FACE DETECTED
    # =====================================================
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark
        # =================================================
        # 8. CALCULATE LEFT & RIGHT EAR
        # =================================================
        left_ear = calculate_ear(
            landmarks,
            LEFT_EYE,
            width,
            height
        )
        right_ear = calculate_ear(
            landmarks,
            RIGHT_EYE,
            width,
            height
        )
        # Average EAR
        ear = (left_ear + right_ear) / 2.0
        # =================================================
        # 9. DISPLAY EAR
        # =================================================
        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        # =================================================
        # 10. EYE BLINK / EYE CLOSURE DETECTION
        # =================================================
        if ear < EAR_THRESHOLD:
            # Start timer when eyes first close
            if eye_closed_start is None:
                eye_closed_start = time.time()
            closed_time = time.time() - eye_closed_start
            cv2.putText(
                frame,
                f"Eyes Closed: {closed_time:.1f}s",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            # =================================================
            # 11. DROWSINESS DETECTION
            # =================================================
            if closed_time >= DROWSINESS_TIME:
                cv2.putText(
                    frame,
                    "DROWSINESS DETECTED!",
                    (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3
                )
                # =================================================
                # 12. AUDIO ALARM
                # =================================================
                if not alarm_on:
                    if alarm is not None:
                        alarm.play(-1)
                    alarm_on = True
        else:
            # Eyes opened again
            eye_closed_start = None
            # Stop alarm
            if alarm_on:
                if alarm is not None:
                    alarm.stop()
                alarm_on = False
            cv2.putText(
                frame,
                "Eyes Open / Normal",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
    # =====================================================
    # 13. NO FACE
    # =====================================================
    else:
        cv2.putText(
            frame,
            "Face Not Detected",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
    # =====================================================
    # 14. DISPLAY WINDOW
    # =====================================================
    cv2.imshow(
        "Driver Drowsiness Detection",
        frame
    )
    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# =========================================================
# 15. CLEANUP
# =========================================================
if alarm_on and alarm is not None:
    alarm.stop()
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
pygame.mixer.quit()
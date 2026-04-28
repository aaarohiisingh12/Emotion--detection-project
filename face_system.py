# face_system_live_clean.py
import cv2
import mediapipe as mp
import math
import pandas as pd
import os

# -------------------------------
# Distance function
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# -------------------------------
# Rule-based emotion detection
def detect_emotion(landmarks):
    try:
        mouth_width = distance(landmarks[61], landmarks[291])
        mouth_open = distance(landmarks[13], landmarks[14])
        left_eyebrow = distance(landmarks[159], landmarks[145])
        right_eyebrow = distance(landmarks[386], landmarks[374])

        if mouth_width > 60:
            return "Happy"
        elif mouth_open > 30:
            return "Surprised"
        elif left_eyebrow < 15 or right_eyebrow < 15:
            return "Angry"
        else:
            return "Neutral"
    except:
        return "Unknown"

# -------------------------------
# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# -------------------------------
# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

# -------------------------------
# Logging
os.makedirs("results", exist_ok=True)
log = []
frame_count = 0

# -------------------------------
# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    landmarks_dict = {}

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # -------------------------------
            # Draw only key landmarks
            key_points = [13, 14, 61, 291, 145, 159, 374, 386]
            for idx in key_points:
                x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                landmarks_dict[idx] = (x, y)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # -------------------------------
            # Optional: draw minimal connections
            cv2.line(frame, landmarks_dict[61], landmarks_dict[291], (0, 255, 0), 2)   # mouth width
            cv2.line(frame, landmarks_dict[13], landmarks_dict[14], (0, 255, 0), 2)    # mouth open
            cv2.line(frame, landmarks_dict[159], landmarks_dict[145], (0, 255, 0), 2)  # left eyebrow
            cv2.line(frame, landmarks_dict[386], landmarks_dict[374], (0, 255, 0), 2)  # right eyebrow

            # -------------------------------
            # Emotion detection
            emotion = detect_emotion(landmarks_dict)
            cv2.putText(frame, f'Emotion: {emotion}',
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

            # Log data
            log.append({"frame": frame_count, "emotion": emotion})

    else:
        cv2.putText(frame, "No face detected",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)

    # Show frame
    cv2.imshow("Emotion Detection (Live)", frame)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# -------------------------------
# Cleanup
cap.release()
cv2.destroyAllWindows()

# Save CSV log
df = pd.DataFrame(log)
df.to_csv("results/emotion_log_live.csv", index=False)
print("DONE. Data saved in results/emotion_log_live.csv")
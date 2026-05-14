import cv2
import mediapipe as mp
import math
import pandas as pd
import os

# -------------------------------
# distance function between 2 points refer to documentation before changes
# math.hypot gives euclidean distance
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# -------------------------------
# emotion detection function
# basic facial measurements and thresholds
# refer to mediapipe documentation for values and changes
def detect_emotion(landmarks):
    try:
        # getting mouth width and how open the mouth is
        mouth_width = distance(landmarks[61], landmarks[291])
        mouth_open = distance(landmarks[13], landmarks[14])

        # measuring how open the eyes are
        left_eye_open = distance(landmarks[159], landmarks[145])
        right_eye_open = distance(landmarks[386], landmarks[374])
        eye_open = (left_eye_open + right_eye_open) / 2

        #eyebrow movement
        left_eyebrow = distance(landmarks[159], landmarks[145])
        right_eyebrow = distance(landmarks[386], landmarks[374])
        eyebrow_avg = (left_eyebrow + right_eyebrow) / 2

        # calculate eye shape ratio (height vs width)
        eye_width = distance(landmarks[33], landmarks[133])
        eye_height = distance(landmarks[159], landmarks[145])
        eye_ratio = eye_height / eye_width

        # nose width (not fully sure how useful this is yet) its the flair for angriness
        nose_width = distance(landmarks[98], landmarks[327])

        # using face width to normalize values so different face sizes work
        face_width = distance(landmarks[234], landmarks[454])

        mouth_width /= face_width
        mouth_open /= face_width
        eye_open /= face_width
        eyebrow_avg /= face_width
        nose_width /= face_width

        # -------------------------------
        # trying a score-based system for angry emotion exp 4
        # instead of just 1 condition

        angry_score = 0

        if eye_open < 0.022:
            angry_score += 1

        if mouth_open < 0.03:
            angry_score += 1

        if eyebrow_avg < 0.03:
            angry_score += 1

        if nose_width < 0.04:
            angry_score += 1

        # if at least 2 conditions match, classify as angry
        if angry_score >= 2:
            return "angry"

        # -------------------------------
        # happy expression (wide smile + open eyes)
        elif mouth_width > 0.5 and eye_open > 0.025:
            return "happy"

        # surprised expression (wide open mouth + eyes)
        elif mouth_open > 0.05 and eye_open > 0.04 and eye_ratio > 0.35:
            return "surprised"

        # sad expression (smaller mouth, less open)
        elif mouth_width < 0.35 and mouth_open < 0.025:
            return "sad"

        else:
            return "neutral"

    except Exception as e:
        print("Error:", e)
        return "unknown"

# -------------------------------
# setting up mediapipe face mesh
# i followed documentation for this part
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,  # since i am using images not video
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# -------------------------------
# folder paths
dataset_path = "dataset"
results_folder = "results"

# making results folder if it doesn't exist
os.makedirs(results_folder, exist_ok=True)

# -------------------------------
# storing results in a list 
log = []

# -------------------------------
# looping through dataset folders
# each folder is named as an emotion
for emotion_folder in os.listdir(dataset_path):
    emotion_dir = os.path.join(dataset_path, emotion_folder)

    if not os.path.isdir(emotion_dir):
        continue

    # looping through all images inside that emotion folder
    for image_name in os.listdir(emotion_dir):
        image_path = os.path.join(emotion_dir, image_name)

        image = cv2.imread(image_path)
        if image is None:
            continue

        # converting BGR to RGB because mediapipe expects RGB its more stable
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        landmarks_dict = {}
        predicted = "no_face"

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # extracting only important landmark points
            for idx in [13, 14, 61, 291, 145, 159, 374, 386, 33, 133, 98, 327, 234, 454]:
                x = int(face_landmarks.landmark[idx].x * image.shape[1])
                y = int(face_landmarks.landmark[idx].y * image.shape[0])
                landmarks_dict[idx] = (x, y)

            predicted = detect_emotion(landmarks_dict)

        # storing results
        log.append({
            "image": image_name,
            "actual": emotion_folder.lower(),
            "predicted": predicted.lower()
        })

        print(f"{image_name} → Actual: {emotion_folder} | Predicted: {predicted}")

# -------------------------------
# saving results to csv file
df = pd.DataFrame(log)
csv_path = os.path.join(results_folder, "batch_results.csv")
df.to_csv(csv_path, index=False)

print("\nDONE. Results saved at:", csv_path)
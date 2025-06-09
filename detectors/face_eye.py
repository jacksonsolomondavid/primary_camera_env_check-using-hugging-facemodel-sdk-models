import cv2
import mediapipe as mp

def detect_face_gaze(image_path):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return False  # No face detected

    face = results.multi_face_landmarks[0]
    # Simplified: just checks if both eyes are detected
    left_eye = [33, 133]
    right_eye = [362, 263]
    landmarks = [face.landmark[i] for i in left_eye + right_eye]

    if all(landmarks):
        return True  # Eyes detected
    return False

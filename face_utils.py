# face_utils.py 
import mediapipe as mp
import cv2
import numpy as np

mp_detect = mp.solutions.face_detection
mp_mesh   = mp.solutions.face_mesh

def crop_face(img_path):
    """
    Crop face from image and resize to 224x224 for CNN input.
    Returns numpy array or None if no face found.
    """
    img = cv2.imread(img_path)
    if img is None: return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    with mp_detect.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as det:
        res = det.process(rgb)
        if not res.detections: return None
        box = res.detections[0].location_data.relative_bounding_box
        pad = 0.12
        x1 = max(0, int((box.xmin - pad) * w))
        y1 = max(0, int((box.ymin - pad) * h))
        x2 = min(w, int((box.xmin + box.width + pad) * w))
        y2 = min(h, int((box.ymin + box.height + pad) * h))
        face = img[y1:y2, x1:x2]
        return cv2.resize(face, (224, 224))

LANDMARK_IDS = {
    'left_temple'  : 234,
    'right_temple' : 454,
    'left_eye_out' : 33,
    'right_eye_out': 263,
    'nose_bridge'  : 6,
    'chin'         : 152,
}

def get_landmarks(img_path):
    """Return pixel coordinates of 6 face landmarks."""
    img = cv2.imread(img_path)
    if img is None: return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    with mp_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                         min_detection_confidence=0.5) as mesh:
        res = mesh.process(rgb)
        if not res.multi_face_landmarks: return None
        lm = res.multi_face_landmarks[0].landmark
        return {k: (int(lm[v].x * w), int(lm[v].y * h))
                for k, v in LANDMARK_IDS.items()}
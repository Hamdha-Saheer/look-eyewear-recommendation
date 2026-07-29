import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# Load model and Haar Cascade for face detection
model = tf.keras.models.load_model('model/model_male.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classes = ['heart', 'oblong', 'oval', 'round', 'square']

def analyze_face_fixed(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect face
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        print("No face detected! Try a clearer photo.")
        return

    # Crop to the first face found
    (x, y, w, h) = faces[0]
    face_crop = img[y:y+h, x:x+w]
    
    # Resize to the model's expected input size
    face_resized = cv2.resize(face_crop, (224, 224))
    
    
    # Convert from BGR (OpenCV default) to RGB (Model expectation)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)   
    
    # Preprocess (Scales pixels to [-1, 1] for ResNetV2)
    arr = preprocess_input(np.expand_dims(face_rgb.astype('float32'), 0))
    
    
    # Predict
    preds = model.predict(arr, verbose=0)[0]
    
    print(f"\n--- CROPPED Results for {image_path} ---")
    for i, p in enumerate(preds):
        print(f"{classes[i].upper()}: {p*100:.2f}%")

analyze_face_fixed('male.jpeg')
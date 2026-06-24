# app.py — FULLY CONVERTED TO PYTORCH
#
# Changes implemented:
# 1. Removed TensorFlow / Keras entirely.
# 2. Added PyTorch model initialization and structural block loading.
# 3. Switched image preprocessing to torchvision.transforms (ImageNet scale matching).
# 4. Kept existing business logic, route handlers, and JSON formats untouched.
#
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS

import torch
import torch.nn as nn
from torchvision import models as tv_models, transforms
from PIL import Image

import numpy as np, json, os, cv2
from werkzeug.utils import secure_filename
from face_utils import crop_face, get_landmarks
from try_on import overlay_glasses

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
ALLOWED = {'png', 'jpg', 'jpeg', 'webp'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. Load class orders saved during training ───────────────────────────────────
male_classes   = list(json.load(open('model/classes_male.json')).keys())
female_classes = list(json.load(open('model/classes_female.json')).keys())
print('Male classes:   ', male_classes)
print('Female classes:', female_classes)
classes_map = {'male': male_classes, 'female': female_classes}

# ── 2. Helper function to initialize model graph matching train_model.py ─────────
def load_pytorch_model(weight_path, num_classes=5):
    """Rebuilds the exact MobileNetV2 architecture variant used in training."""
    model = tv_models.mobilenet_v2(pretrained=False) # Architecture layout only
    
    # Rebuild the exact custom classifier block from your training file
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(model.last_channel),
        nn.Dropout(p=0.5), 
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(p=0.4), 
        nn.Linear(256, num_classes)
    )
    
    # Load weights map directly onto the active processing device
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval() # Toggle evaluation mode to lock BatchNorm and Dropout layers
    return model

print('Loading PyTorch models...')
models = {
    'male'  : load_pytorch_model('model/model_male.pth'),
    'female': load_pytorch_model('model/model_female.pth')
}

recs = json.load(open('recommendations.json'))
print('Ready!')

# ── 3. PyTorch Preprocessing Pipeline (CRITICAL: Matches ImageNet statistics) ─────
predict_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ── Build glasses index ───────────────────────────────────────────────────────
def build_glasses_index():
    index = {}
    glasses_dir = 'static/glasses'
    if not os.path.exists(glasses_dir):
        print('WARNING: static/glasses/ folder not found!')
        return index
    for f in sorted(os.listdir(glasses_dir)):
        if not f.endswith('.png'):
            continue
        parts = f.replace('.png', '').rsplit('_', 1)
        if len(parts) == 2:
            style = parts[0]
            if style not in index:
                index[style] = []
            index[style].append(f)
    return index

glasses_index = build_glasses_index()
print('Glasses index:', {k: len(v) for k, v in glasses_index.items()})

def allowed(fn):
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in ALLOWED

# ── 4. Updated Prediction Routine ─────────────────────────────────────────────
def run_prediction(face_bgr, gender):
    """
    Run PyTorch inference on a cropped face image matrix.
    face_bgr: numpy array in BGR format (from cv2/crop_face)
    Returns: (shape_str, confidence_float, all_preds_array)
    """
    # Convert BGR -> RGB to match PIL / PyTorch training data orientation
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert NumPy array to PIL Image so torchvision transforms accept it cleanly
    pil_img = Image.fromarray(face_rgb)
    
    # Apply standard ImageNet transform matrix and add batch dimension [1, 3, 224, 224]
    tensor_img = predict_transforms(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = models[gender](tensor_img)
        # Apply Softmax function to translate raw logits to probabilities [0, 1]
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # Extract prediction elements back into numpy space for processing
    preds = probabilities.cpu().numpy()
    idx = int(np.argmax(preds))
    
    shape = classes_map[gender][idx]
    conf = round(float(preds[idx]) * 100, 1)
    
    return shape, conf, preds

def build_recommendations(shape, gender, preds):
    shape_recs  = recs.get(shape.lower(), {}).get(gender, {})
    best_styles = shape_recs.get('best', [])

    recommendations_with_images = []
    for item in best_styles:
        style = item['style']
        imgs  = glasses_index.get(style, [])
        recommendations_with_images.append({
            'style' : style,
            'name'  : item['name'],
            'reason': item['reason'],
            'images': imgs
        })

    # Top 2 predictions for low-confidence fallback warning systems
    top2 = [{'shape': classes_map[gender][i].lower(),
             'pct':   round(float(preds[i]) * 100, 1)}
            for i in np.argsort(preds)[::-1][:2]]

    return {
        'description'    : shape_recs.get('description', ''),
        'tip'            : shape_recs.get('tip', ''),
        'avoid'          : shape_recs.get('avoid', []),
        'recommendations': recommendations_with_images,
        'top2'           : top2,
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form.get('gender', 'male')
    file   = request.files.get('image')

    if not file or not allowed(file.filename):
        return jsonify({'error': 'Please upload a JPG or PNG image'}), 400

    path = f'uploads/{secure_filename(file.filename)}'
    file.save(path)

    # 1. Detect and crop face
    face = crop_face(path)
    if face is None:
        os.remove(path)
        return jsonify({'error': 'No face detected. Please use a clear front-facing photo.'}), 400

    # 2. Get landmarks for AR overlay
    landmarks = get_landmarks(path)

    # 3. Predict face shape using the updated PyTorch engine
    shape, conf, preds = run_prediction(face, gender)

    # 4. Build recommendations
    rec_data = build_recommendations(shape, gender, preds)

    os.remove(path)

    return jsonify({
        'shape'          : shape.lower(),
        'confidence'     : conf,
        'low_confidence' : conf < 70,
        'landmarks'      : landmarks,
        **rec_data
    })

@app.route('/predict_shape', methods=['POST'])
def predict_shape():
    data   = request.get_json()
    shape  = data.get('shape', '').lower()
    gender = data.get('gender', 'male')

    shape_recs  = recs.get(shape, {}).get(gender, {})
    best_styles = shape_recs.get('best', [])

    recommendations_with_images = []
    for item in best_styles:
        style = item['style']
        imgs  = glasses_index.get(style, [])
        recommendations_with_images.append({
            'style' : style,
            'name'  : item['name'],
            'reason': item['reason'],
            'images': imgs
        })

    return jsonify({
        'shape'          : shape,
        'description'    : shape_recs.get('description', ''),
        'tip'            : shape_recs.get('tip', ''),
        'avoid'          : shape_recs.get('avoid', []),
        'recommendations': recommendations_with_images,
    })

@app.route('/tryon', methods=['POST'])
def tryon():
    style     = request.form.get('style')
    img_file  = request.form.get('img_file')
    landmarks = json.loads(request.form.get('landmarks'))
    file      = request.files.get('image')

    if not file:
        return jsonify({'error': 'No image provided'}), 400

    path   = f'uploads/tryon_{secure_filename(file.filename)}'
    output = f'uploads/result_{style}.jpg'
    file.save(path)

    if not img_file:
        available = glasses_index.get(style, [])
        img_file  = available[0] if available else f'{style}_1.png'

    if not overlay_glasses(path, img_file, landmarks, output):
        os.remove(path)
        return jsonify({'error': 'Overlay failed — check glasses PNG file'}), 500

    os.remove(path)
    return send_file(output, mimetype='image/jpeg')

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=5000)
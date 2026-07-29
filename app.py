# app.py — FULLY CORRECTED PYTORCH VERSION
#
# Fixes applied:
# 1. Pure PyTorch — no TensorFlow at all
# 2. Classifier head EXACTLY matches train_model.py
# 3. Classes loaded correctly from JSON (index->name mapping)
# 4. BGR->RGB + ImageNet normalisation (matches training)
# 5. /predict_shape route added
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
print(f"Using device: {device}")

# ── Load class orders ─────────────────────────────────────────────────────────
# train_model.py saves: {"heart": 0, "oblong": 1, "oval": 2, "round": 3, "square": 4}
# We need index->name: {0: "heart", 1: "oblong", 2: "oval", 3: "round", 4: "square"}

def load_classes(path):
    """Load class JSON and return both name->idx and idx->name mappings."""
    name_to_idx = json.load(open(path))
    idx_to_name = {v: k for k, v in name_to_idx.items()}
    return idx_to_name

male_classes   = load_classes('model/classes_male.json')
female_classes = load_classes('model/classes_female.json')
classes_map    = {'male': male_classes, 'female': female_classes}

print('Male classes:  ', male_classes)
print('Female classes:', female_classes)

# ── Load PyTorch models ───────────────────────────────────────────────────────
def load_pytorch_model(weight_path, num_classes=5):
    """Build EXACT same architecture as train_model.py and load weights."""
    model = tv_models.mobilenet_v2(weights=None)   # architecture only

    # MUST match train_model.py classifier exactly
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(model.last_channel),
        nn.Dropout(p=0.5),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(256, num_classes)
    )
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model

print('Loading PyTorch models...')
models_dict = {
    'male'  : load_pytorch_model('model/model_male.pth'),
    'female': load_pytorch_model('model/model_female.pth')
}
recs = json.load(open('recommendations.json'))
print('Models ready!')

# ── Preprocessing — MUST match train_model.py eval_transforms ────────────────
predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# ── Glasses index ─────────────────────────────────────────────────────────────
def build_glasses_index():
    index = {}
    gdir = 'static/glasses'
    if not os.path.exists(gdir):
        print('WARNING: static/glasses/ not found!')
        return index
    for f in sorted(os.listdir(gdir)):
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
print('Glasses:', {k: len(v) for k, v in glasses_index.items()})

def allowed(fn):
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in ALLOWED

# ── Prediction ────────────────────────────────────────────────────────────────
def run_prediction(face_bgr, gender):
    """
    Predict face shape from cropped face (BGR numpy array from cv2).
    Returns (shape_string, confidence_float, preds_numpy_array)
    """
    # Step 1: BGR -> RGB  (cv2 loads BGR, PIL/PyTorch expects RGB)
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    # Step 2: numpy array -> PIL Image
    pil_img = Image.fromarray(face_rgb)

    # Step 3: apply transforms -> add batch dimension [1, 3, 224, 224]
    tensor = predict_transform(pil_img).unsqueeze(0).to(device)

    # Step 4: inference
    with torch.no_grad():
        outputs = models_dict[gender](tensor)
        probs   = torch.nn.functional.softmax(outputs, dim=1)[0]

    preds = probs.cpu().numpy()
    idx   = int(np.argmax(preds))

    # Step 5: map index -> class name using saved class order
    shape = classes_map[gender][idx]   # e.g. {0:'heart', 1:'oblong', ...}[2] = 'oval'
    conf  = round(float(preds[idx]) * 100, 1)

    return shape, conf, preds

def build_recommendations(shape, gender, preds):
    shape_recs = recs.get(shape.lower(), {}).get(gender, {})
    best = []
    for item in shape_recs.get('best', []):
        best.append({
            'style' : item['style'],
            'name'  : item['name'],
            'reason': item['reason'],
            'images': glasses_index.get(item['style'], [])
        })

    # Top 2 predictions (for low-confidence fallback)
    top2 = [
        {
            'shape': classes_map[gender][i].lower(),
            'pct'  : round(float(preds[i]) * 100, 1)
        }
        for i in np.argsort(preds)[::-1][:2]
    ]

    return {
        'description'    : shape_recs.get('description', ''),
        'tip'            : shape_recs.get('tip', ''),
        'avoid'          : shape_recs.get('avoid', []),
        'recommendations': best,
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

    # 1. Crop face with MediaPipe
    face = crop_face(path)
    if face is None:
        os.remove(path)
        return jsonify({
            'error': 'No face detected. Use a clear front-facing photo.'
        }), 400

    # 2. Get landmarks for AR try-on
    landmarks = get_landmarks(path)

    # 3. Predict
    shape, conf, preds = run_prediction(face, gender)

    # 4. Build recommendations
    rec_data = build_recommendations(shape, gender, preds)

    os.remove(path)

    return jsonify({
        'shape'         : shape.lower(),
        'confidence'    : conf,
        'low_confidence': conf < 70,
        'landmarks'     : landmarks,
        **rec_data
    })


@app.route('/predict_shape', methods=['POST'])
def predict_shape():
    """Called when user manually selects their face shape."""
    data   = request.get_json()
    shape  = data.get('shape', '').lower()
    gender = data.get('gender', 'male')

    shape_recs = recs.get(shape, {}).get(gender, {})
    best = []
    for item in shape_recs.get('best', []):
        best.append({
            **item,
            'images': glasses_index.get(item['style'], [])
        })

    return jsonify({
        'shape'          : shape,
        'description'    : shape_recs.get('description', ''),
        'tip'            : shape_recs.get('tip', ''),
        'avoid'          : shape_recs.get('avoid', []),
        'recommendations': best,
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
        return jsonify({'error': 'Overlay failed — check glasses PNG'}), 500

    os.remove(path)
    return send_file(output, mimetype='image/jpeg')


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=5000)
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import numpy as np
import traceback

# ============================================================
# OncoScan V2 FINAL — Flask Backend
# ============================================================
# Model: EfficientNet-B4 @ 456×456 + Metadata Fusion
# Classes: 8 (MEL, NV, BCC, AK, BKL, DF, VASC, SCC)
# Input: Image + Age + Sex + Anatomical Site
# ============================================================

app = Flask(__name__)
CORS(app)

# ---- Configuration from model config ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
CONFIG_PATH = os.path.join(MODEL_DIR, 'oncoscan_v2_final_config.json')

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

IMG_SIZE = config['img_size']  # 456
NUM_CLASSES = config['num_classes']  # 8
CLASS_NAMES = config['class_names']
CLASS_FULL_NAMES = config['class_full_names']
MALIGNANT_INDICES = config['malignant_indices']  # [0, 2, 3, 7]
BENIGN_INDICES = config['benign_indices']  # [1, 4, 5, 6]
SITE_CATEGORIES = config['site_categories']
META_COLS = config['meta_cols']

# ---- Risk info for each class ----
CLASS_INFO = {
    'MEL': {
        'risk': 'HIGH',
        'severity': 'Malignant',
        'description': 'Melanoma is the most dangerous form of skin cancer. It develops from melanocytes and can spread to other organs.',
        'precautions': [
            'Seek immediate consultation with a dermatologist.',
            'Avoid sun exposure and use SPF 50+ sunscreen.',
            'Do NOT scratch, pick, or irritate the lesion.',
            'A biopsy is strongly recommended for confirmation.'
        ]
    },
    'NV': {
        'risk': 'LOW',
        'severity': 'Benign',
        'description': 'Melanocytic Nevus (common mole) is a benign growth of melanocytes. Most moles are harmless.',
        'precautions': [
            'Monitor for changes in size, shape, or color (ABCDE rule).',
            'Use sunscreen to prevent new moles from forming.',
            'Reassess if the mole becomes asymmetric or has irregular borders.'
        ]
    },
    'BCC': {
        'risk': 'MODERATE',
        'severity': 'Malignant',
        'description': 'Basal Cell Carcinoma is the most common type of skin cancer. It rarely spreads but can damage surrounding tissue.',
        'precautions': [
            'Consult a dermatologist for evaluation.',
            'Treatment is usually effective if caught early.',
            'Protect the area from further sun damage.',
            'Surgical excision may be recommended.'
        ]
    },
    'AK': {
        'risk': 'MODERATE',
        'severity': 'Pre-cancerous',
        'description': 'Actinic Keratosis is a pre-cancerous condition caused by UV damage. It can develop into squamous cell carcinoma if untreated.',
        'precautions': [
            'Consult a dermatologist for treatment options.',
            'Apply prescribed topical treatments as directed.',
            'Strict sun protection is essential.',
            'Regular follow-up to monitor for progression.'
        ]
    },
    'BKL': {
        'risk': 'LOW',
        'severity': 'Benign',
        'description': 'Benign Keratosis (seborrheic keratosis) is a common, harmless skin growth that appears with aging.',
        'precautions': [
            'Usually no treatment required.',
            'Consult a doctor if it becomes irritated or changes rapidly.',
            'Can be removed for cosmetic reasons if desired.'
        ]
    },
    'DF': {
        'risk': 'LOW',
        'severity': 'Benign',
        'description': 'Dermatofibroma is a common, harmless firm bump in the skin, usually on the legs.',
        'precautions': [
            'No treatment usually needed.',
            'Consult a doctor if it grows rapidly or becomes painful.',
            'Surgical removal is an option if bothersome.'
        ]
    },
    'VASC': {
        'risk': 'LOW',
        'severity': 'Benign',
        'description': 'Vascular Lesion includes conditions like cherry angiomas and pyogenic granulomas. Most are benign.',
        'precautions': [
            'Generally harmless, monitor for changes.',
            'Consult a doctor if bleeding occurs frequently.',
            'Laser treatment available if cosmetically undesirable.'
        ]
    },
    'SCC': {
        'risk': 'HIGH',
        'severity': 'Malignant',
        'description': 'Squamous Cell Carcinoma is the second most common skin cancer. It can spread if not treated early.',
        'precautions': [
            'Seek prompt dermatological consultation.',
            'Biopsy recommended for definitive diagnosis.',
            'Treatment usually involves surgical excision.',
            'Regular follow-up for recurrence monitoring.'
        ]
    }
}

# ---- Load Model ----
print("Loading OncoScan V2 FINAL model...")
model = None
try:
    import tensorflow as tf

    # Register custom objects needed to load the model
    @tf.keras.utils.register_keras_serializable(package='OncoScan')
    class ModalityDropout(tf.keras.layers.Layer):
        def __init__(self, rate=0.5, **kwargs):
            super().__init__(**kwargs)
            self.rate = rate

        def call(self, inputs, training=None):
            if training:
                mask = tf.cast(
                    tf.random.uniform([tf.shape(inputs)[0], 1]) > self.rate,
                    inputs.dtype
                )
                return inputs * mask
            return inputs

        def get_config(self):
            config = super().get_config()
            config.update({'rate': self.rate})
            return config

    @tf.keras.utils.register_keras_serializable(package='OncoScan')
    class FocalLoss(tf.keras.losses.Loss):
        def __init__(self, gamma=2.0, label_smoothing=0.0, **kwargs):
            super().__init__(**kwargs)
            self.gamma = gamma
            self.label_smoothing = label_smoothing

        def call(self, y_true, y_pred):
            num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
            if self.label_smoothing > 0:
                y_true = y_true * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            cross_entropy = -y_true * tf.math.log(y_pred)
            focal_weight = tf.pow(1.0 - y_pred, self.gamma)
            loss = focal_weight * cross_entropy
            return tf.reduce_sum(loss, axis=-1)

        def get_config(self):
            config = super().get_config()
            config.update({
                'gamma': self.gamma,
                'label_smoothing': self.label_smoothing,
            })
            return config

    model_path = os.path.join(MODEL_DIR, 'oncoscan_v2_final_best.keras')
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded: {model_path}")
        print(f"   Input: image {IMG_SIZE}×{IMG_SIZE} + {len(META_COLS)} metadata features")
        print(f"   Output: {NUM_CLASSES} classes")
    else:
        print(f"⚠️ Model file not found: {model_path}")
        print("   Place 'oncoscan_v2_final_best.keras' in the model/ directory.")

except Exception as e:
    print(f"❌ Failed to load model: {e}")
    traceback.print_exc()


# ---- Image Preprocessing ----
def preprocess_image(image_path):
    """Load and resize image to match training pipeline."""
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("cv2.imread returned None — invalid image file")
        # Resize to model input size (456×456)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # Convert BGR to RGB (cv2 loads as BGR, model trained on RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        return np.expand_dims(img, axis=0)
    except Exception as e:
        print(f"Image preprocessing failed: {e}")
        return None


# ---- Metadata Preprocessing ----
def preprocess_metadata(age, sex, body_site='unknown'):
    """Encode metadata exactly as training pipeline does."""
    try:
        # Age normalization (same as training: divide by 90, clip 0-1)
        age_norm = min(max(float(age) / 90.0, 0.0), 1.0)

        # Sex encoding
        sex_lower = sex.lower() if sex else ''
        if sex_lower == 'male':
            sex_enc = 1.0
        elif sex_lower == 'female':
            sex_enc = 0.0
        else:
            sex_enc = 0.5  # unknown/other

        # Anatomical site one-hot encoding
        site_map = {
            'anterior_torso': 'anterior_torso',
            'head_neck': 'head_neck', 'head/neck': 'head_neck',
            'lower_extremity': 'lower_extremity',
            'posterior_torso': 'posterior_torso',
            'upper_extremity': 'upper_extremity',
            'other': 'other',
        }
        site_clean = site_map.get(body_site.lower().replace(' ', '_'), 'unknown')
        site_features = [1.0 if cat == site_clean else 0.0 for cat in SITE_CATEGORIES]

        # Combine: [age_norm, sex_enc, site_0, site_1, ..., site_6]
        metadata = [age_norm, sex_enc] + site_features
        return np.array([metadata], dtype=np.float32)

    except Exception as e:
        print(f"Metadata preprocessing failed: {e}")
        return None


# ---- Routes ----
@app.route('/')
def home():
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/assets/<filename>')
def serve_assets(filename):
    return send_from_directory(ASSETS_DIR, filename)


@app.route('/<filename>')
def serve_file(filename):
    return send_from_directory(BASE_DIR, filename)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Place model file in model/ directory.'}), 503

        # Get form data
        age = request.form.get('age', '55')
        sex = request.form.get('gender', 'unknown')
        body_site = request.form.get('body_site', 'unknown')
        image_file = request.files.get('image')

        if not image_file:
            return jsonify({'error': 'No image provided'}), 400

        print(f"🔍 Prediction request — Age: {age}, Sex: {sex}, Site: {body_site}")

        # Save uploaded image temporarily
        filename = secure_filename(image_file.filename)
        os.makedirs('uploads', exist_ok=True)
        filepath = os.path.join('uploads', filename)
        image_file.save(filepath)

        # Preprocess
        image = preprocess_image(filepath)
        metadata = preprocess_metadata(age, sex, body_site)

        # Clean up
        os.remove(filepath)

        if image is None or metadata is None:
            return jsonify({'error': 'Failed to process input data'}), 400

        # Predict
        predictions = model.predict(
            {'image_input': image, 'meta_input': metadata},
            verbose=0
        )[0]  # Shape: (8,)

        # Get top prediction
        pred_class_idx = int(np.argmax(predictions))
        pred_class = CLASS_NAMES[pred_class_idx]
        confidence = float(predictions[pred_class_idx])

        # Calculate malignant probability
        malignant_prob = float(sum(predictions[i] for i in MALIGNANT_INDICES))
        benign_prob = float(sum(predictions[i] for i in BENIGN_INDICES))
        is_malignant = malignant_prob >= 0.5

        # Get class info
        info = CLASS_INFO[pred_class]

        # Build top-3 predictions
        top3_indices = np.argsort(predictions)[::-1][:3]
        top3 = [
            {
                'class': CLASS_NAMES[i],
                'name': CLASS_FULL_NAMES[CLASS_NAMES[i]],
                'probability': round(float(predictions[i]) * 100, 1),
                'severity': CLASS_INFO[CLASS_NAMES[i]]['severity']
            }
            for i in top3_indices
        ]

        response = {
            'prediction': {
                'class': pred_class,
                'name': CLASS_FULL_NAMES[pred_class],
                'confidence': round(confidence * 100, 1),
                'risk': info['risk'],
                'severity': info['severity'],
                'description': info['description'],
            },
            'malignant_probability': round(malignant_prob * 100, 1),
            'benign_probability': round(benign_prob * 100, 1),
            'is_malignant': is_malignant,
            'top3': top3,
            'precautions': info['precautions'],
            'result': f"{'⚠️ Potentially Malignant' if is_malignant else '✅ Likely Benign'} — "
                      f"{CLASS_FULL_NAMES[pred_class]} ({confidence*100:.1f}% confidence)",
            'score': round(confidence, 4),
        }

        print(f"   Result: {pred_class} ({CLASS_FULL_NAMES[pred_class]}) — "
              f"{confidence*100:.1f}% — {'MALIGNANT' if is_malignant else 'BENIGN'}")

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'Internal Server Error', 'details': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 OncoScan running at http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)

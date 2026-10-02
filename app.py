import os
import json
import tempfile
import traceback
import numpy as np
import gradio as gr
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

# Force TensorFlow to use CPU to prevent ZeroGPU hanging
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
    import spaces
    GPU_DECORATOR = spaces.GPU
except ImportError:
    GPU_DECORATOR = lambda fn: fn  # no-op locally

# ============================================================
# OncoScan V2 FINAL — Backend
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
CONFIG_PATH = os.path.join(MODEL_DIR, 'oncoscan_v2_final_config.json')

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

IMG_SIZE = config['img_size']
NUM_CLASSES = config['num_classes']
CLASS_NAMES = config['class_names']
CLASS_FULL_NAMES = config['class_full_names']
MALIGNANT_INDICES = config['malignant_indices']
BENIGN_INDICES = config['benign_indices']
SITE_CATEGORIES = config['site_categories']

CLASS_INFO = {
    'MEL': { 'risk': 'HIGH', 'severity': 'Malignant', 'description': 'Melanoma is dangerous.', 'precautions': ['Seek dermatologist immediately.'] },
    'NV': { 'risk': 'LOW', 'severity': 'Benign', 'description': 'Common mole.', 'precautions': ['Monitor for changes.'] },
    'BCC': { 'risk': 'MODERATE', 'severity': 'Malignant', 'description': 'Basal Cell Carcinoma.', 'precautions': ['Consult a dermatologist.'] },
    'AK': { 'risk': 'MODERATE', 'severity': 'Pre-cancerous', 'description': 'Actinic Keratosis.', 'precautions': ['Consult a dermatologist.'] },
    'BKL': { 'risk': 'LOW', 'severity': 'Benign', 'description': 'Benign Keratosis.', 'precautions': ['Usually no treatment required.'] },
    'DF': { 'risk': 'LOW', 'severity': 'Benign', 'description': 'Dermatofibroma.', 'precautions': ['Usually no treatment required.'] },
    'VASC': { 'risk': 'LOW', 'severity': 'Benign', 'description': 'Vascular Lesion.', 'precautions': ['Monitor for changes.'] },
    'SCC': { 'risk': 'HIGH', 'severity': 'Malignant', 'description': 'Squamous Cell Carcinoma.', 'precautions': ['Seek dermatological consultation.'] }
}

# ---- Load Model ----
print("Loading OncoScan model...")
model = None
try:
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package='OncoScan')
    class ModalityDropout(tf.keras.layers.Layer):
        def __init__(self, rate=0.5, **kwargs):
            super().__init__(**kwargs)
            self.rate = rate
        def call(self, inputs, training=None):
            if training:
                mask = tf.cast(tf.random.uniform([tf.shape(inputs)[0], 1]) > self.rate, inputs.dtype)
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
            return tf.reduce_sum(focal_weight * cross_entropy, axis=-1)
        def get_config(self):
            config = super().get_config()
            config.update({'gamma': self.gamma, 'label_smoothing': self.label_smoothing})
            return config

    model_path = os.path.join(MODEL_DIR, 'oncoscan_v2_final_best.keras')
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"[SUCCESS] Model loaded")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")

def preprocess_image(image_path):
    import cv2
    img = cv2.imread(image_path)
    if img is None: return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img.astype(np.float32), axis=0)

def preprocess_metadata(age, sex, body_site='unknown'):
    age_norm = min(max(float(age) / 90.0, 0.0), 1.0)
    sex_lower = str(sex).lower() if sex else ''
    sex_enc = 1.0 if sex_lower == 'male' else (0.0 if sex_lower == 'female' else 0.5)
    site_map = {
        'anterior_torso': 'anterior_torso', 'head_neck': 'head_neck',
        'head/neck': 'head_neck', 'lower_extremity': 'lower_extremity',
        'posterior_torso': 'posterior_torso', 'upper_extremity': 'upper_extremity',
        'other': 'other',
    }
    site_clean = site_map.get(str(body_site).lower().replace(' ', '_'), 'unknown')
    site_features = [1.0 if cat == site_clean else 0.0 for cat in SITE_CATEGORIES]
    metadata = [age_norm, sex_enc] + site_features
    return np.array([metadata], dtype=np.float32)

def run_prediction(image_path, age, gender):
    if model is None: return {'error': 'Model not loaded'}
    image = preprocess_image(image_path)
    metadata = preprocess_metadata(age, gender)
    if image is None or metadata is None: return {'error': 'Invalid input'}

    predictions = model.predict({'image_input': image, 'meta_input': metadata}, verbose=0)[0]

    pred_class_idx = int(np.argmax(predictions))
    pred_class = CLASS_NAMES[pred_class_idx]
    confidence = float(predictions[pred_class_idx])
    malignant_prob = float(sum(predictions[i] for i in MALIGNANT_INDICES))
    benign_prob = float(sum(predictions[i] for i in BENIGN_INDICES))
    is_malignant = malignant_prob >= 0.5
    info = CLASS_INFO[pred_class]

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

    return {
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
    }

# ---- FastAPI App ----
api = FastAPI()
api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@api.get("/health")
async def health():
    return {"status": "ok"}

@api.post("/predict")
async def predict_api(
    image: UploadFile = File(...),
    age: str = Form("55"),
    gender: str = Form("unknown")
):
    try:
        contents = await image.read()
        suffix = os.path.splitext(image.filename or "img.jpg")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        result = run_prediction(tmp_path, age, gender)
        os.unlink(tmp_path)
        return result
    except Exception as e:
        return {"error": str(e)}

# ---- Gradio App for HF Spaces Detection ----
@GPU_DECORATOR
def gradio_predict(image, age, gender):
    import cv2
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        cv2.imwrite(tmp.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        tmp_path = tmp.name
    res = run_prediction(tmp_path, str(age), gender)
    os.unlink(tmp_path)
    return res

demo = gr.Interface(
    fn=gradio_predict,
    inputs=[gr.Image(type="numpy"), gr.Number(value=30), gr.Dropdown(["male", "female", "other"], value="male")],
    outputs=gr.JSON(),
)

app = gr.mount_gradio_app(api, demo, path="/")

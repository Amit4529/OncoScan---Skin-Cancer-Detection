import os
import json
import tempfile
import traceback
import numpy as np
import gradio as gr
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

# ZeroGPU support (only available on HF Spaces)
try:
    import spaces
    GPU_DECORATOR = spaces.GPU
except ImportError:
    GPU_DECORATOR = lambda fn: fn  # no-op locally

# ============================================================
# OncoScan V2 FINAL — Gradio + FastAPI Backend (HF Spaces)
# ============================================================
# Model: EfficientNet-B4 @ 456x456 + Metadata Fusion
# Classes: 8 (MEL, NV, BCC, AK, BKL, DF, VASC, SCC)
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
META_COLS = config['meta_cols']

CLASS_INFO = {
    'MEL': {
        'risk': 'HIGH', 'severity': 'Malignant',
        'description': 'Melanoma is the most dangerous form of skin cancer. It develops from melanocytes and can spread to other organs.',
        'precautions': ['Seek immediate consultation with a dermatologist.', 'Avoid sun exposure and use SPF 50+ sunscreen.', 'Do NOT scratch, pick, or irritate the lesion.', 'A biopsy is strongly recommended for confirmation.']
    },
    'NV': {
        'risk': 'LOW', 'severity': 'Benign',
        'description': 'Melanocytic Nevus (common mole) is a benign growth of melanocytes. Most moles are harmless.',
        'precautions': ['Monitor for changes in size, shape, or color (ABCDE rule).', 'Use sunscreen to prevent new moles from forming.', 'Reassess if the mole becomes asymmetric or has irregular borders.']
    },
    'BCC': {
        'risk': 'MODERATE', 'severity': 'Malignant',
        'description': 'Basal Cell Carcinoma is the most common type of skin cancer. It rarely spreads but can damage surrounding tissue.',
        'precautions': ['Consult a dermatologist for evaluation.', 'Treatment is usually effective if caught early.', 'Protect the area from further sun damage.', 'Surgical excision may be recommended.']
    },
    'AK': {
        'risk': 'MODERATE', 'severity': 'Pre-cancerous',
        'description': 'Actinic Keratosis is a pre-cancerous condition caused by UV damage. It can develop into squamous cell carcinoma if untreated.',
        'precautions': ['Consult a dermatologist for treatment options.', 'Apply prescribed topical treatments as directed.', 'Strict sun protection is essential.', 'Regular follow-up to monitor for progression.']
    },
    'BKL': {
        'risk': 'LOW', 'severity': 'Benign',
        'description': 'Benign Keratosis (seborrheic keratosis) is a common, harmless skin growth that appears with aging.',
        'precautions': ['Usually no treatment required.', 'Consult a doctor if it becomes irritated or changes rapidly.', 'Can be removed for cosmetic reasons if desired.']
    },
    'DF': {
        'risk': 'LOW', 'severity': 'Benign',
        'description': 'Dermatofibroma is a common, harmless firm bump in the skin, usually on the legs.',
        'precautions': ['No treatment usually needed.', 'Consult a doctor if it grows rapidly or becomes painful.', 'Surgical removal is an option if bothersome.']
    },
    'VASC': {
        'risk': 'LOW', 'severity': 'Benign',
        'description': 'Vascular Lesion includes conditions like cherry angiomas and pyogenic granulomas. Most are benign.',
        'precautions': ['Generally harmless, monitor for changes.', 'Consult a doctor if bleeding occurs frequently.', 'Laser treatment available if cosmetically undesirable.']
    },
    'SCC': {
        'risk': 'HIGH', 'severity': 'Malignant',
        'description': 'Squamous Cell Carcinoma is the second most common skin cancer. It can spread if not treated early.',
        'precautions': ['Seek prompt dermatological consultation.', 'Biopsy recommended for definitive diagnosis.', 'Treatment usually involves surgical excision.', 'Regular follow-up for recurrence monitoring.']
    }
}

# ---- Load Model ----
print("Loading OncoScan V2 FINAL model...")
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
        print(f"[SUCCESS] Model loaded: {model_path}")
    else:
        print(f"[WARNING] Model not found: {model_path}")

except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    traceback.print_exc()


# ---- Preprocessing ----
def preprocess_image(image_path):
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img.astype(np.float32), axis=0)


def preprocess_metadata(age, sex, body_site='unknown'):
    age_norm = min(max(float(age) / 90.0, 0.0), 1.0)
    sex_lower = sex.lower() if sex else ''
    sex_enc = 1.0 if sex_lower == 'male' else (0.0 if sex_lower == 'female' else 0.5)
    site_map = {
        'anterior_torso': 'anterior_torso', 'head_neck': 'head_neck',
        'head/neck': 'head_neck', 'lower_extremity': 'lower_extremity',
        'posterior_torso': 'posterior_torso', 'upper_extremity': 'upper_extremity',
        'other': 'other',
    }
    site_clean = site_map.get(body_site.lower().replace(' ', '_'), 'unknown')
    site_features = [1.0 if cat == site_clean else 0.0 for cat in SITE_CATEGORIES]
    metadata = [age_norm, sex_enc] + site_features
    return np.array([metadata], dtype=np.float32)


def run_prediction(image_path, age, gender, body_site='unknown'):
    """Core prediction logic shared by both Gradio UI and REST API."""
    if model is None:
        return {'error': 'Model not loaded'}

    image = preprocess_image(image_path)
    metadata = preprocess_metadata(age, gender, body_site)
    if image is None or metadata is None:
        return {'error': 'Failed to process input'}

    predictions = model.predict(
        {'image_input': image, 'meta_input': metadata}, verbose=0
    )[0]

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

    result_str = "Potentially Malignant" if is_malignant else "Likely Benign"
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
        'result': f"{result_str} - {CLASS_FULL_NAMES[pred_class]} ({confidence*100:.1f}% confidence)",
        'score': round(confidence, 4),
    }


# ---- FastAPI app with custom /predict endpoint ----
api = FastAPI(title="OncoScan API")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None, "version": "v2-final"}


@api.post("/predict")
async def predict_api(
    image: UploadFile = File(...),
    age: str = Form("55"),
    gender: str = Form("unknown"),
    body_site: str = Form("unknown"),
):
    try:
        contents = await image.read()
        suffix = os.path.splitext(image.filename or "img.jpg")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        result = run_prediction(tmp_path, age, gender, body_site)
        os.unlink(tmp_path)

        if 'error' in result:
            return {"error": result['error']}
        return result
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ---- Gradio UI (shown on HF Space homepage) ----
@GPU_DECORATOR
def gradio_predict(image, age, gender):
    """Gradio interface wrapper."""
    if image is None:
        return {"error": "Please upload an image"}
    # Save numpy image from Gradio to temp file
    import cv2
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        cv2.imwrite(tmp.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        tmp_path = tmp.name

    result = run_prediction(tmp_path, str(age), gender)
    os.unlink(tmp_path)
    return result


with gr.Blocks(title="OncoScan - Skin Cancer Detection") as demo:
    gr.Markdown("# OncoScan - Skin Cancer Detection API")
    gr.Markdown("Upload a skin lesion image for AI-powered screening.")
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="numpy", label="Skin Lesion Image")
            age_input = gr.Number(value=30, label="Age", minimum=1, maximum=120)
            gender_input = gr.Dropdown(["male", "female", "other"], value="male", label="Gender")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            output = gr.JSON(label="Analysis Results")
    btn.click(fn=gradio_predict, inputs=[img_input, age_input, gender_input], outputs=output)
    gr.Markdown("---")
    gr.Markdown("*This is a screening tool only. Always consult a dermatologist for diagnosis.*")

# Mount Gradio onto FastAPI (keeps /predict and /health routes working)
app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

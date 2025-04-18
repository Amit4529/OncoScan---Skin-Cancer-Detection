from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
import cv2
import gdown
import traceback

# Initialize Flask app with root directory for everything
app = Flask(__name__)
CORS(app)

IMG_SIZE = (128, 128)

# Step 1: Download models if not present
models_info = {
    "best_skin_cancer_recall.h5": "1asWlGYmajSTFYGCSMPBbxRplAcCp5NRH",
    "best_skin_cancer_auc.h5": "19fXD76gaTdrHIqe44fwYgYAQ9-zycJlt"
}

for filename, file_id in models_info.items():
    if not os.path.exists(filename):
        print(f"Downloading {filename} from Google Drive...")
        gdown.download(f'https://drive.google.com/uc?id={file_id}', filename, quiet=False)

# Step 2: Load models
def load_model_safe(path):
    if os.path.exists(path):
        try:
            return tf.keras.models.load_model(path)
        except Exception as e:
            print(f"Error loading {path}: {str(e)}")
    return None

model_auc = load_model_safe('best_skin_cancer_auc.h5')
model_recall = load_model_safe('best_skin_cancer_recall.h5')
print(f"Models loaded - AUC: {model_auc is not None}, Recall: {model_recall is not None}")

# Image preprocessing
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("cv2.imread returned None")
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0
        return np.expand_dims(img.astype(np.float32), axis=0)
    except Exception as e:
        print(f"Image preprocessing failed: {str(e)}")
        return None

# Metadata preprocessing
def preprocess_metadata(age, gender):
    try:
        age = float(age) / 100.0
        gender_encoded = 1.0 if gender.lower() == 'male' else 0.0
        diagnosis_placeholder = 0.0
        return np.array([[age, gender_encoded, diagnosis_placeholder]], dtype=np.float32)
    except Exception as e:
        print(f"Metadata preprocessing failed: {str(e)}")
        return None

# User-friendly result
def get_user_friendly_result(label, confidence):
    print(f"Label: {label}, Confidence: {confidence}")
    if confidence>0.65 :
        return {
            "result": "There are signs that this could be a malignant lesion. Please consult a certified dermatologist immediately. (Cancerous)",
            "precautions": [
                "Avoid sun exposure and use broad-spectrum sunscreen.",
                "Do not scratch or irritate the lesion.",
                "Seek medical attention for biopsy and clinical diagnosis."
            ]
        }
    else:
        return {
            "result": "This lesion is likely benign. Regular monitoring is recommended (Non-Cancerous).",
            "precautions": [
                "Avoid prolonged sun exposure.",
                "Reassess with a dermatologist if the lesion changes shape, color, or size.",
                "Stay hydrated and follow a skin care routine."
            ]
        }

# Routes
@app.route('/')
def home():
    # Serve the index.html file from the root directory
    return send_from_directory(os.getcwd(), 'index.html')

# Serve static files (CSS, JS, images, etc.) from the root directory
@app.route('/<filename>')
def serve_file(filename):
    return send_from_directory(os.getcwd(), filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = request.form.get('age')
        gender = request.form.get('gender')
        image_file = request.files.get('image')

        if not image_file:
            return jsonify({'error': 'No image provided'}), 400

        print(f"Received - Age: {age}, Gender: {gender}, Image: {image_file.filename}")

        filename = secure_filename(image_file.filename)
        os.makedirs('uploads', exist_ok=True)
        filepath = os.path.join('uploads', filename)
        image_file.save(filepath)

        image = preprocess_image(filepath)
        metadata = preprocess_metadata(age, gender)

        os.remove(filepath)

        if image is None or metadata is None:
            return jsonify({'error': 'Failed to process input data'}), 400

        predictions = []
        if model_auc:
            predictions.append(model_auc.predict([image, metadata])[0][0])
        if model_recall:
            predictions.append(model_recall.predict([image, metadata])[0][0])

        if not predictions:
            return jsonify({'error': 'No models available for prediction'}), 500

        avg_pred = np.mean(predictions)
        label = "Malignant" if avg_pred >= 0.5 else "Benign"
        confidence = avg_pred if avg_pred >= 0.5 else 1 - avg_pred

        response = get_user_friendly_result(label, confidence)
        # response["score"] = round(float(confidence), 4)
        response["score"] = round(float(confidence), 4) if not np.isnan(confidence) else None

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'Internal Server Error', 'details': str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))


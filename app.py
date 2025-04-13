from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__, static_folder='static', template_folder='docs')
CORS(app)

IMG_SIZE = (128, 128)

# Load both models safely
def load_model_safe(path):
    if os.path.exists(path):
        try:
            return tf.keras.models.load_model(path)
        except Exception as e:
            print(f"Error loading {path}: {str(e)}")
    return None

model_auc = load_model_safe('best_skin_cancer_auc.h5')
model_recall = load_model_safe('best_skin_cancer_recall.h5')

def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0
        return np.expand_dims(img.astype(np.float32), axis=0)
    except Exception as e:
        print(f"Image preprocessing failed: {str(e)}")
        return None

def preprocess_metadata(age, gender):
    try:
        age = float(age) / 100.0
        gender_encoded = 1.0 if gender.lower() == 'male' else 0.0
        diagnosis_placeholder = 0.0
        return np.array([[age, gender_encoded, diagnosis_placeholder]], dtype=np.float32)
    except:
        return None

def get_user_friendly_result(label, confidence):
    if label == 'Malignant':
        return {
            "result": "There are signs that this could be a malignant lesion. Please consult a certified dermatologist immediately. (Cancerous)",
            "precautions": [
                "Avoid sun exposure and use broad-spectrum sunscreen.",
                "Do not scratch or irritate the lesion.",
                "Seek medical attention for biopsy and clinical diagnosis."
            ]
        }
    elif confidence < 0.65:
        return {
            "result": "This lesion is likely benign, but it may carry some risk. Regular monitoring is recommended (Non-Cancerous).",
            "precautions": [
                "Avoid prolonged sun exposure.",
                "Reassess with a dermatologist if the lesion changes shape, color, or size.",
                "Stay hydrated and follow a skin care routine."
            ]
        }
    else:
        return {
            "result": "The lesion appears benign and does not currently show signs of skin cancer.",
            "precautions": [
                "Still, it's wise to monitor skin changes regularly.",
                "Use sun protection, especially during peak hours.",
                "Maintain regular dermatological checkups."
            ]
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = request.form['age']
        gender = request.form['gender']
        image_file = request.files['image']

        if not image_file:
            return jsonify({'error': 'No image provided'}), 400

        filename = secure_filename(image_file.filename)
        filepath = os.path.join('uploads', filename)
        os.makedirs('uploads', exist_ok=True)
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
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

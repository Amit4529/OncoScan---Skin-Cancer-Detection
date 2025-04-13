from flask import Flask, request, jsonify, send_file

from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from data_utils import load_image  # Your custom function

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Allow frontend requests

MODEL_PATH = 'cancer_detection_model.h5'
IMG_SIZE = 224

model = load_model(MODEL_PATH)  # Load model once

def preprocess_metadata(age, gender):
    sex_male = 1.0 if gender.lower() == 'male' else 0.0
    sex_unknown = 1.0 if gender.lower() == 'unknown' else 0.0
    age = float(age)
    return np.array([[age, sex_male, sex_unknown]], dtype='float32')

@app.route('/')
def home():
    return send_file('index.html')

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

        image = load_image(filepath)
        image = tf.expand_dims(image, axis=0)
        metadata = preprocess_metadata(age, gender)

        prediction = model.predict([image, metadata])[0][0]
        os.remove(filepath)

        result = {
            'score': float(prediction),
            'result': 'High risk of skin cancer. Please consult a dermatologist.' if prediction >= 0.5 else 'Low risk of skin cancer. Still, regular checkups are important.'
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

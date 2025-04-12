import tensorflow as tf
import numpy as np
import os
from data_utils import load_image  # Reuse your existing function
from tensorflow.keras.models import load_model

IMG_SIZE = 224
MODEL_PATH = 'cancer_detection_model.h5'

def preprocess_metadata(age, gender):
    # Normalize gender into one-hot encoding: 'sex_male' and 'sex_unknown'
    sex_male = 1.0 if gender.lower() == 'male' else 0.0
    sex_unknown = 1.0 if gender.lower() == 'unknown' else 0.0
    age = float(age)
    return np.array([[age, sex_male, sex_unknown]], dtype='float32')

def predict(image_path, age, gender):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Load and preprocess image
    image = load_image(image_path)
    image = tf.expand_dims(image, axis=0)  # Add batch dimension

    # Preprocess tabular data
    metadata = preprocess_metadata(age, gender)

    # Load model
    model = load_model(MODEL_PATH)

    # Predict
    prediction = model.predict([image, metadata])[0][0]
    percentage = prediction * 100
    print(f"\nPrediction Confidence: {percentage:.2f}%")

    if prediction >= 0.41:
     print("Based on the analysis, there's a risk of skin cancer.\nPlease consult a dermatologist for a professional diagnosis.")
    else:
     print("Based on the analysis, the risk of skin cancer is *low*.\nStill, regular checkups are important.")
        
if __name__ == "__main__":
    # User input
    img_path = input("Enter image path: ")
    age_input = input("Enter age: ")
    gender_input = input("Enter gender (male, female, unknown): ")

    predict(img_path, age_input, gender_input)

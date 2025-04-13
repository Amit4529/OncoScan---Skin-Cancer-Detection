import tensorflow as tf
import numpy as np
import cv2
import os

def preprocess_image(image_path, img_size=(128, 128)):
    """Preprocess a single image to match model input."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        img = cv2.resize(img, img_size)
        img = img / 255.0  # Normalize
        return img.astype(np.float32)
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def preprocess_metadata(age, gender):
    """Preprocess age and gender to match model input."""
    try:
        age_normalized = float(age) / 100.0
        gender_encoded = 1 if gender.lower() == 'male' else 0
        diagnosis_placeholder = 0
        return np.array([[age_normalized, gender_encoded, diagnosis_placeholder]], dtype=np.float32)
    except Exception as e:
        print(f"Error processing metadata: {str(e)}")
        return None

def load_model_safely(model_path):
    """Load model if it exists, else return None."""
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Error loading model {model_path}: {str(e)}")
    return None

def predict_ensemble(image_path, age, gender):
    """Make an ensemble prediction using both models (AUC & Recall)."""
    model_auc = load_model_safely('best_skin_cancer_auc.h5')
    model_recall = load_model_safely('best_skin_cancer_recall.h5')

    if not model_auc and not model_recall:
        raise FileNotFoundError("Model files missing: 'best_skin_cancer_auc.h5' or 'best_skin_cancer_recall.h5'.")

    image = preprocess_image(image_path)
    meta = preprocess_metadata(age, gender)

    if image is None or meta is None:
        return None

    image = np.expand_dims(image, axis=0)  # Shape: (1, 128, 128, 3)
    predictions = []

    if model_auc:
        try:
            pred_auc = model_auc.predict([image, meta])[0][0]
            predictions.append(pred_auc)
        except Exception as e:
            print(f"Prediction failed for AUC model: {str(e)}")

    if model_recall:
        try:
            pred_recall = model_recall.predict([image, meta])[0][0]
            predictions.append(pred_recall)
        except Exception as e:
            print(f"Prediction failed for Recall model: {str(e)}")

    if not predictions:
        return None

    avg_pred = np.mean(predictions)
    label = 'Malignant' if avg_pred > 0.5 else 'Benign'
    confidence = avg_pred if avg_pred > 0.5 else 1 - avg_pred
    return label, confidence

def explain_result(label, confidence):
    """Return a human-readable explanation and precautions."""
    if label == "Malignant":
        message = (
            "⚠️ The result suggests a high risk of skin cancer (Malignant).\n\n"
            "Please consult a certified dermatologist as soon as possible for confirmation and further diagnosis.\n\n"
            "**Important Steps:**\n"
            "- Avoid self-diagnosis or using home remedies.\n"
            "- Do not ignore changes in the skin lesion.\n"
            "- Schedule an appointment with a healthcare provider.\n"
        )
    else:
        if confidence < 0.75:
            message = (
                "ℹ️ The result suggests the lesion is likely not cancerous (Benign), but there is some uncertainty.\n\n"
                "**Caution:** There may still be a risk of developing skin cancer.\n"
                "We strongly advise monitoring the lesion and visiting a dermatologist for a proper evaluation.\n"
            )
        else:
            message = (
                "✅ The lesion appears to be non-cancerous (Benign).\n"
                "However, regular self-checks and yearly dermatology visits are still recommended.\n"
            )

    message += (
        "\n**General Precautions to Prevent Skin Cancer:**\n"
        "- Use sunscreen with SPF 30+ regularly.\n"
        "- Avoid prolonged sun exposure, especially between 10 AM – 4 PM.\n"
        "- Wear protective clothing and hats outdoors.\n"
        "- Avoid tanning beds and artificial UV exposure.\n"
        "- Check your skin monthly for new or changing moles/lesions.\n"
    )

    return message

def main():
    """Interactive CLI - For debugging or local use."""
    print("Skin Cancer Prediction Tool")
    image_path = input("Enter the path to the image (e.g., path/to/image.jpg): ")
    if not os.path.exists(image_path):
        print("❌ Error: Image file does not exist.")
        return

    age = input("Enter your age: ")
    gender = input("Enter your gender (male/female): ")

    result = predict_ensemble(image_path, age, gender)
    if result:
        label, confidence = result
        explanation = explain_result(label, confidence)
        print("\n--- Prediction Result ---")
        print(explanation)
    else:
        print("Prediction failed. Please check the input and try again.")

if __name__ == "__main__":
    main()

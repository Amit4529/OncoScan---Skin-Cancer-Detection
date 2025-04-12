# 🧬 OncoScan - Skin Cancer Detection using Deep Learning

> **"Early Detection Saves Lives"**  

> OncoScan leverages the power of deep learning to detect skin cancer using dermatoscopic images and patient metadata for accurate, real-time diagnosis.

---

## 🚀 Project Overview

**OncoScan** is a hybrid skin cancer detection system that combines **Convolutional Neural Networks (CNNs)** for dermatoscopic image analysis with **tabular 
metadata** (age, gender, and anatomical site) to enhance classification accuracy.

🎯 The goal is to support **early and reliable detection** of malignant skin lesions, potentially saving lives through timely diagnosis.

---

## 📊 Dataset: HAM10000

- **Full Name**: *Human Against Machine with 10,000 training images*
- **Images**: 10,015 high-resolution dermatoscopic images
- **Metadata**:
  - Patient **Age**
  - Patient **Sex**
  - **Anatomical Site** of the lesion
- **Binary Labels**:
  - 🟥 **Malignant (1)** → `mel`, `bcc`, `akiec`
  - 🟩 **Benign (0)** → All other classes

---

## 🧠 Model Architecture

### 🔹 Image Pathway (CNN)
- ResNet50 (pretrained on ImageNet, frozen layers)
- Global Average Pooling
- Dense Layer → 128 units

### 🔹 Metadata Pathway (Tabular)
- Dense Layer → 16 units

### 🔹 Fusion & Output
- Concatenation of CNN & metadata outputs
- Dense Layer → 64 units
- **Output Layer** → Sigmoid activation (Binary Classification)

---

## 🛠️ Project Structure

ONCOSCAN---SKIN-CANCER-DETECTION/

│

├── app.py                       # Flask app (main entry)

├── main.py                      # Additional routing or logic

├── model.py                     # Model definition/loading

├── data_utils.py                # Data preprocessing utilities
├── cancer_detection_model1.h5   # Trained model
├── output.py                    # Output handling logic (if needed)
│
├── templates/
│   └── index.html               # HTML UI template
│
├── static/
│   ├── styles.css               # CSS file
│   ├── script.js                # JavaScript file
│   ├── logo.png                 # Web interface logo
│   └── 06-skin-cancer-moles.png # Sample UI image
│
├── README.md                    # Project documentation

Installation Requirements: Before running the project, install the necessary Python packages:
•	TensorFlow
•	Pandas
•	NumPy
•	scikit-learn
•	matplotlib (optional for visualization)

pip install tensorflow pandas numpy scikit-learn matplotlib

----

## 🧠 Model Architecture

### 🔹 Image Pathway (CNN)
- ResNet50 pretrained on ImageNet (frozen)
- Global Average Pooling
- Dense Layer: 128 units

### 🔹 Metadata Pathway (Tabular)
- Dense Layer: 16 units

### 🔹 Fusion & Output
- Concatenation of image and tabular features
- Dense Layer: 64 units
- Final Output: Sigmoid activation (Binary Classification)

---

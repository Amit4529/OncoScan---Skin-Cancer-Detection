# 🧬 OncoScan - Skin Cancer Detection using Deep Learning

> **"Early Detection Saves Lives"**  

> OncoScan leverages the power of deep learning to detect skin cancer using dermatoscopic images and patient metadata for accurate, real-time diagnosis.

---

## 🚀 Project Overview

**OncoScan** is a hybrid skin cancer detection system that combines **Convolutional Neural Networks (CNNs)** for dermatoscopic image analysis with **tabular 
metadata** (age, gender, and anatomical site) to enhance classification accuracy.

Current Accuracy - 73-74%

Server is not connected due to memory limit exceeded on render/replit platforms, but the modal is completely ready

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
![image](https://github.com/user-attachments/assets/d744a2fe-508b-4a61-88ba-1b49a5003798)
![image](https://github.com/user-attachments/assets/cb96ec07-c79f-483c-8f5c-7de555c81a81)
![image](https://github.com/user-attachments/assets/4b2a4624-55fa-430c-9f4a-8198ff5718f3)
![Screenshot_13-4-2025_15320_127 0 0 1](https://github.com/user-attachments/assets/ff29ba2a-304f-4657-a962-6055b25e199f)



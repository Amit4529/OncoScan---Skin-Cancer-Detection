# 🧬 OncoScan - Skin Cancer Detection using Deep Learning

> **"Early Detection Saves Lives"**  

> OncoScan leverages the power of deep learning to detect skin cancer using dermatoscopic images and patient metadata for accurate, real-time diagnosis.

---

## 🚀 Project Overview

**OncoScan** is a hybrid skin cancer detection system that combines **Convolutional Neural Networks (CNNs)** for dermatoscopic image analysis with **tabular 
metadata** (age, gender, and anatomical site) to enhance classification accuracy.

Current Accuracy - 73-74%

Video Link:- https://drive.google.com/file/d/1zsE8II9oWUNXoy6Bw5OccfVCHBoPtUpL/view  (there was error in video we submitted)

made some commit after 4:30 just because of server deploy due to large memory (but now everything is done everything is been fixed)

🎯 The goal is to support **early and reliable detection** of malignant skin lesions, potentially saving lives through timely diagnosis.

---



## 📊 Dataset: BNC20000
Full Name: Binary Neoplastic Classification with 20,000 dermatoscopic images
Images: 20,000 high-resolution dermatoscopic images
Metadata:

Patient Age
Image
Patient Sex
Binary Labels:
🟥 Malignant (1) → Includes types like melanoma, basal cell carcinoma, and actinic keratosis
🟩 Benign (0) → Includes all other non-cancerous skin lesion types

---

## 🧠 Model Architecture
🔹 Image Pathway (CNN)

Input: 128×128×3 dermatoscopic image

Conv2D Layers: 32 → 64 → 128 → 256 filters

Each layer: ReLU Activation, Batch Normalization, MaxPooling

Flatten Layer at the end

🔹 Metadata Pathway (Tabular)

Input: 3 features (Age, Sex, Diagnosis_1)

Dense Layer → 32 units with ReLU

Dropout (0.3) for regularization

🔹 Fusion & Output

Concatenation of Image & Metadata branches

Dense → 512 → Dropout (0.5)

Dense → 256 → Dropout (0.3)

Output Layer → 1 unit with Sigmoid activation

Optimizer: Adam

Loss: Binary Crossentropy

Metrics: Accuracy, AUC, Precision, Recall
---

## 🛠️ Project Structure

ONCOSCAN---SKIN-CANCER-DETECTION/

│

├── app.py                       # Flask app (main entry)

├── main.py                      # Additional routing or logic

├── model.py                     # Model definition/loading

├── data_preprocessing.py                # Data preprocessing utilities


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



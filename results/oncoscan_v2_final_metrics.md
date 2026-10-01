# 📊 OncoScan V2 FINAL — Model Metrics Report

**Model**: EfficientNet-B4 @ 456×456 + Metadata Fusion  
**Dataset**: ISIC 2019 (25,331 images, 8 classes)  
**Training**: Oversampling (min 1500/class) + Focal Loss (γ=2.0, smoothing=0.02)  
**Evaluation**: Test-Time Augmentation (8 transforms)  

---

## Overall Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **88.2%** |
| **AUC (macro)** | **0.9757** |
| **AUC (weighted)** | **0.9769** |
| **F1 (macro)** | **0.8365** |
| **F1 (weighted)** | **0.8806** |
| **Binary Accuracy** | **91.9%** |
| **Binary AUC** | **0.9709** |
| **Malignant Recall** | **85.7%** |

---

## Per-Class Metrics

| Class | Full Name | Precision | Recall | F1-Score | Support |
|-------|-----------|-----------|--------|----------|---------|
| MEL | Melanoma | 0.8435 | 0.7703 | 0.8052 | 679 |
| NV | Melanocytic Nevus | 0.9149 | 0.9461 | 0.9302 | 1931 |
| BCC | Basal Cell Carcinoma | 0.9104 | 0.9158 | 0.9131 | 499 |
| AK | Actinic Keratosis | 0.7542 | 0.6846 | 0.7177 | 130 |
| BKL | Benign Keratosis | 0.7835 | 0.8473 | 0.8142 | 393 |
| DF | Dermatofibroma | 0.9355 | 0.8056 | 0.8657 | 36 |
| VASC | Vascular Lesion | 0.9444 | 0.8947 | 0.9189 | 38 |
| SCC | Squamous Cell Carcinoma | 0.8451 | 0.6383 | 0.7273 | 94 |

### Averages

| Type | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| Macro avg | 0.8664 | 0.8128 | 0.8365 |
| Weighted avg | 0.8812 | 0.8821 | 0.8806 |

---

## Binary Classification (Malignant vs Benign)

**Malignant**: MEL, BCC, AK, SCC  
**Benign**: NV, BKL, DF, VASC  

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Benign | 0.9197 | 0.9558 | 0.9374 | 2398 |
| Malignant | 0.9190 | 0.8573 | 0.8871 | 1402 |

---

## Per-Class AUC

| Class | AUC |
|-------|-----|
| MEL (Melanoma) | 0.960 |
| NV (Melanocytic Nevus) | 0.977 |
| BCC (Basal Cell Carcinoma) | 0.993 |
| AK (Actinic Keratosis) | 0.954 |
| BKL (Benign Keratosis) | 0.978 |
| DF (Dermatofibroma) | 0.988 |
| VASC (Vascular Lesion) | 0.987 |
| SCC (Squamous Cell Carcinoma) | 0.941 |

---

## Confidence Thresholding

| Threshold | Accuracy | Coverage |
|-----------|----------|----------|
| 0.3 | 88.3% | 99.9% |
| 0.4 | 88.6% | 99.1% |
| 0.5 | 90.1% | 95.4% |
| 0.6 | 93.6% | 85.9% |
| 0.7 | 95.9% | 74.6% |
| 0.8 | 97.4% | 55.3% |

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Backbone | EfficientNet-B4 (ImageNet pretrained) |
| Input Resolution | 456 × 456 |
| Batch Size | 12 |
| Phase 1 | 5 epochs (frozen backbone), LR=1e-3 |
| Phase 2 | 40 epochs (top 50% unfrozen), LR=1e-4→1e-6 |
| LR Schedule | 3-epoch warmup → cosine annealing |
| Early Stopping | patience=12 on val_auc (triggered at epoch 39) |
| Best Epoch | 27 (val_auc=0.9874) |
| Loss | Focal Loss (γ=2.0, label_smoothing=0.02) |
| Optimizer | AdamW (weight_decay=1e-5, clipnorm=1.0) |
| Imbalance Strategy | Oversampling (min 1500 samples/class) |
| Metadata | age, sex, anatomical site (9 features) |
| TTA | 8 transforms (flips, rotations, combinations) |

## Model Files

| File | Description |
|------|-------------|
| `oncoscan_v2_final_best.keras` | Best checkpoint (epoch 27) |
| `oncoscan_v2_final.keras` | Final model |
| `oncoscan_v2_final_config.json` | Config for inference |
| `oncoscan_v2_final_results.json` | Full metrics JSON |
| `oncoscan_v2_final_weights.weights.h5` | Weights only |

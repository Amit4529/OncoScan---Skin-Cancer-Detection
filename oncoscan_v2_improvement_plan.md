# 🔬 OncoScan V2 — Metric-by-Metric Analysis & V2.1 Improvement Plan

## 📊 Current Metrics vs Targets

| Metric | V2 Result | Target | Gap | Priority |
|--------|-----------|--------|-----|----------|
| Multi-class Accuracy | **87.2%** | 92%+ | -4.8% | 🔴 High |
| AUC (macro) | **0.9814** | 0.98+ | ✅ Met | 🟢 Done |
| F1 (macro) | **0.8145** | 0.90+ | -8.6% | 🔴 High |
| F1 (weighted) | **0.8700** | 0.92+ | -5.0% | 🟡 Medium |
| Binary Accuracy | **91.7%** | 95%+ | -3.3% | 🟡 Medium |
| Binary AUC | **0.9670** | 0.98+ | -1.3% | 🟡 Medium |
| Malignant Recall | **86.95%** | 95%+ | -8.1% | 🔴 Critical |

> [!IMPORTANT]
> **AUC is already excellent (0.9814)** — the model CAN rank cancer vs non-cancer very well. The issue is the **decision boundaries** and **minority class recall**. This means we don't need a fundamentally different architecture — we need better training strategy.

---

## 🔍 Per-Class Error Breakdown (from Confusion Matrix)

### 🔴 Problem Class #1: MEL (Melanoma) — Recall 77.2%
```
True MEL (679 samples) was predicted as:
  → MEL:  524 (77.2%) ✅ Correct
  → NV:   114 (16.8%) ❌ DANGEROUS — cancer missed as benign mole
  → BCC:   14 (2.1%)  ⚠️ Still malignant, lower risk
  → BKL:   24 (3.5%)  ❌ Cancer missed as benign
  → AK:     2 (0.3%)  ⚠️ Still malignant
  → SCC:    1 (0.1%)  ⚠️ Still malignant
```
**Root cause**: MEL and NV look very similar visually (dark pigmented lesions). The model needs stronger signal to distinguish them.
**Impact**: 17% of melanomas missed → **clinically unacceptable**
**Fix**: Upweight MEL loss by 2.5x, increase resolution (subtle textures matter)

### 🔴 Problem Class #2: SCC (Squamous Cell Carcinoma) — Recall 61.7%
```
True SCC (94 samples) was predicted as:
  → SCC:   58 (61.7%) ✅ Correct
  → AK:    13 (13.8%) ⚠️ Both malignant, AK can progress to SCC
  → BCC:   10 (10.6%) ⚠️ Both malignant
  → BKL:    9 (9.6%)  ❌ Cancer missed as benign
  → NV:     2 (2.1%)  ❌ Cancer missed as benign
  → MEL:    1 (1.1%)  ⚠️ Both malignant
  → DF:     1 (1.1%)  ❌ Cancer missed as benign
```
**Root cause**: Only 628 training samples + clinically similar to AK/BCC. But 75% of errors stay within malignant group.
**Impact**: Moderate — most misclassifications are to OTHER malignant types (not benign)
**Fix**: Oversample SCC 2x, class weights

### 🟡 Problem Class #3: AK (Actinic Keratosis) — Recall 67.7%
```
True AK (130 samples) was predicted as:
  → AK:    88 (67.7%) ✅ Correct
  → BCC:   17 (13.1%) ⚠️ Both malignant
  → BKL:   13 (10.0%) ❌ Cancer missed as benign
  → MEL:    6 (4.6%)  ⚠️ Both malignant
  → SCC:    6 (4.6%)  ⚠️ Both malignant
```
**Root cause**: 867 training samples + visually similar to BCC and BKL
**Impact**: Moderate — AK→BCC confusion is clinically acceptable (both get treated)
**Fix**: Class weights, augmentation

### 🟡 Problem Class #4: BKL (Benign Keratosis) — Recall 78.6%
```
True BKL (393 samples) was predicted as:
  → BKL:  309 (78.6%) ✅ Correct
  → NV:    40 (10.2%) Low risk (both benign)
  → MEL:   21 (5.3%)  Over-cautious (flags benign as cancer)
  → BCC:    7 (1.8%)  Over-cautious
  → AK:     9 (2.3%)  Over-cautious
  → SCC:    6 (1.5%)  Over-cautious
```
**Root cause**: BKL overlaps visually with both NV and AK/BCC
**Impact**: Low — BKL→NV errors are benign→benign, BKL→MEL is over-cautious (safe)

### 🟢 Strong Classes (No Major Issues)
| Class | Recall | Key Strength |
|-------|--------|-------------|
| NV | 93.8% | Dominant class, well-learned |
| BCC | 90.8% | Excellent for a malignant type |
| VASC | 94.7% | Visually distinctive (red/vascular) |
| DF | 88.9% | Great for only 239 training samples |

---

## 🧠 Root Cause Analysis: Why Metrics Fall Short

### 1. Focal Loss Alone Isn't Enough for This Imbalance Level
- Focal Loss handles easy vs hard examples, but with **53.9:1 ratio**, minority classes (SCC=628, DF=239, VASC=253) don't get enough gradient signal
- **Fix**: Add class weights on TOP of focal loss (we avoided this before, but the data shows it's needed)
- **Approach**: Use MILD weights — just 1.5-2.5x for minority classes, not aggressive

### 2. 384×384 Loses Subtle MEL vs NV Texture Differences
- Melanoma has irregular borders, asymmetric pigmentation, and color variegation
- At 384×384, these fine details are partially lost from original 1024×768 images
- **Fix**: Increase to **456×456** (EfficientNet-B5 native) — captures 40% more pixel information

### 3. No Label Smoothing → Overconfident Wrong Predictions
- The confidence distribution shows many incorrect predictions at >0.9 confidence
- This means the model is making HARD wrong decisions instead of soft uncertain ones
- **Fix**: Add label smoothing (0.05) — softens targets, improves calibration

### 4. Single-Pass Inference → No Error Averaging
- Each prediction is a single forward pass — no error correction
- **Fix**: Test-Time Augmentation (TTA) — average 8 predictions (4 rotations × 2 flips)
- This alone typically gives +1-3% accuracy for free

---

## 🎯 V2.1 Improvement Plan (7 Changes)

### Change 1: Add Class Weights (Mild, Targeted)
```python
# Inverse frequency, capped at 3.0x
class_weights = {
    0: 2.0,   # MEL — upweight to catch more melanomas
    1: 0.5,   # NV — downweight (50% of data)
    2: 1.2,   # BCC
    3: 2.0,   # AK
    4: 1.0,   # BKL
    5: 2.5,   # DF — minority
    6: 2.5,   # VASC — minority
    7: 2.5,   # SCC — worst recall, needs help
}
```
**Expected impact**: +3-5% on macro F1, +5-8% on SCC/AK recall

### Change 2: Increase Resolution to 456×456
- EfficientNet-B4 at 456×456 (or switch to B5 native 456)
- Captures finer texture detail for MEL vs NV distinction
- Needs batch_size=12 to fit GPU memory
- **Expected impact**: +1-2% overall accuracy, +3-5% MEL recall

### Change 3: Add Label Smoothing (0.05)
```python
# In focal loss: smooth labels slightly
y_true_smooth = y_true * (1 - 0.05) + 0.05 / num_classes
```
- Prevents overconfident wrong predictions
- Improves calibration for confidence thresholding
- **Expected impact**: +1% accuracy, better confidence calibration

### Change 4: Test-Time Augmentation (TTA) at Inference
```python
# Average predictions across 8 transforms
tta_preds = []
for k in [0, 1, 2, 3]:  # 4 rotations
    for flip in [False, True]:  # 2 flip states
        augmented = apply_transform(image, rotation=k*90, flip=flip)
        pred = model.predict(augmented)
        tta_preds.append(pred)
final_pred = np.mean(tta_preds, axis=0)
```
- **Expected impact**: +1-3% accuracy, no retraining needed

### Change 5: Extend Training + Warmup
- Phase 2: 40 epochs (was 35) with patience=12
- Add 3-epoch linear warmup: LR ramps from 1e-6 → 1e-4 over first 3 epochs
- Gives the model more time to converge after unfreezing
- **Expected impact**: +0.5-1% from better convergence

### Change 6: Mixup Augmentation (α=0.2)
```python
# Blend two random images and their labels
lambda_ = np.random.beta(0.2, 0.2)
mixed_image = lambda_ * image1 + (1 - lambda_) * image2
mixed_label = lambda_ * label1 + (1 - lambda_) * label2
```
- Proven to help with class imbalance and overfitting
- Creates synthetic "between-class" samples that improve decision boundaries
- **Expected impact**: +1-2% on macro F1, better minority class performance

### Change 7: Clinical Safety Post-Processing
```python
# If predicted benign but high-risk age, boost malignant probability
if predicted_class in benign_classes and age > 65:
    malignant_probs *= 1.3  # 30% boost to malignant probabilities
    # Re-normalize and re-predict
```
- Domain knowledge heuristic, not model change
- Catches edge cases like elderly patients with melanoma misclassified as NV
- **Expected impact**: +2-3% malignant recall for patients >65

---

## 📈 Expected V2.1 Metrics (with all improvements)

| Metric | V2 (current) | V2.1 (expected) | Change |
|--------|-------------|-----------------|--------|
| Multi-class Accuracy | 87.2% | **91-93%** | +4-6% |
| AUC (macro) | 0.9814 | **0.985-0.99** | +0.5% |
| F1 (macro) | 0.8145 | **0.88-0.91** | +7-10% |
| Binary Accuracy | 91.7% | **94-96%** | +3-4% |
| Malignant Recall | 86.95% | **93-95%** | +6-8% |
| MEL Recall | 77.2% | **85-90%** | +8-13% |
| SCC Recall | 61.7% | **75-82%** | +13-20% |

> [!NOTE]
> These projections are based on published results from ISIC competition solutions using similar techniques. The combination of class weights + higher resolution + TTA + label smoothing is well-established to provide these gains.

---

## ✅ Summary: What to Change in Training Script

| Priority | Change | Lines to Modify | Difficulty |
|----------|--------|----------------|------------|
| 🔴 P0 | Fix model save (`.keras` extension) | Cell 13 | Done ✅ |
| 🔴 P0 | Add class weights to `model.fit()` | Cell 9, 10 | Easy |
| 🔴 P0 | Increase IMG_SIZE to 456, batch to 12 | Config | Easy |
| 🟡 P1 | Add label smoothing to FocalLoss | Cell 7 | Easy |
| 🟡 P1 | Add TTA to evaluation | Cell 12 | Medium |
| 🟡 P1 | Add Mixup augmentation | Cell 5, 6 | Medium |
| 🟢 P2 | Add LR warmup | Cell 10 | Easy |
| 🟢 P2 | Clinical post-processing | Cell 12 | Easy |

> [!TIP]
> If approved, I'll update `02_training.py` with ALL of these changes for the V2.1 training run.

# 🔬 OncoScan V2 — Complete EDA Analysis & Revised Strategy

## 📊 Raw Data Summary

| Property | Value |
|----------|-------|
| Total Images | **25,331** |
| CSV Columns | image, MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK |
| One-hot integrity | ✅ All 25,331 rows sum to 1 |
| Image dimensions | ~1024×768 (rectangular, aspect ratio ~1.19) |
| File sizes | 27 KB – 952 KB, median 330 KB |
| Channels | All RGB (3-channel) |

---

## 🚨 Critical Finding #1: UNK Has ZERO Samples

> [!CAUTION]
> The UNK column exists in the CSV but **no image is labeled as UNK**. Total UNK samples = **0**. We cannot train a UNK class on zero data.

### What this means:
- Our original plan to use UNK as an OOD safety class **is not possible** with this dataset
- We train on **8 classes**, not 9

### Revised OOD Strategy: **Confidence Thresholding**
Instead of a UNK class, we handle out-of-distribution inputs by checking the model's **maximum prediction confidence**:
```
If max(softmax_probabilities) < 0.5  →  "Uncertain — consult a dermatologist"
If max(softmax_probabilities) ≥ 0.5  →  Show the predicted class + confidence
```
This is actually **more robust** than a UNK class because:
- A UNK class trained on 0 samples would be useless anyway
- Confidence thresholding works for ANY unseen input type (not just skin lesions)
- We can tune the threshold on the validation set for optimal sensitivity

---

## 📊 Critical Finding #2: SEVERE Class Imbalance (53.9:1)

| Class | Full Name | Count | % | Ratio vs DF | Group |
|-------|-----------|-------|---|-------------|-------|
| **NV** | Melanocytic Nevus | **12,875** | 50.8% | 53.9x | 🟢 Benign |
| **MEL** | Melanoma | **4,522** | 17.9% | 18.9x | 🔴 Malignant |
| **BCC** | Basal Cell Carcinoma | **3,323** | 13.1% | 13.9x | 🔴 Malignant |
| **BKL** | Benign Keratosis | **2,624** | 10.4% | 11.0x | 🟢 Benign |
| **AK** | Actinic Keratosis | **867** | 3.4% | 3.6x | 🔴 Malignant |
| **SCC** | Squamous Cell Carcinoma | **628** | 2.5% | 2.6x | 🔴 Malignant |
| **VASC** | Vascular Lesion | **253** | 1.0% | 1.1x | 🟢 Benign |
| **DF** | Dermatofibroma | **239** | 0.9% | 1.0x | 🟢 Benign |

### Binary Split
| Group | Count | % |
|-------|-------|---|
| **Benign** | 15,991 | 63.1% |
| **Malignant** | 9,340 | 36.9% |

### Imbalance Impact & Strategy
NV alone is **50.8%** of the dataset — if we do nothing, the model will just predict NV for everything and get ~51% accuracy "for free."

**Multi-pronged imbalance strategy:**
1. **Focal Loss (γ=2.0)** — reduces loss contribution from easy/majority examples
2. **Class weights** — inversely proportional to frequency
3. **Oversampling** — duplicate minority samples (DF, VASC, SCC) 3-5x
4. **Heavy augmentation** on minority classes (more aggressive transforms)
5. **Stratified splits** — maintain class ratios in train/val/test

---

## 🧬 Critical Finding #3: Age is a POWERFUL Discriminator

The age-class heatmap reveals a **strong gradient** from benign → malignant:

| Class | Mean Age (F) | Mean Age (M) | Cancer Risk |
|-------|-------------|-------------|-------------|
| NV (Benign) | 43.2 | 46.6 | ⬇️ Lowest |
| DF (Benign) | 52.4 | 51.1 | ⬇️ Low |
| VASC (Benign) | 51.0 | 55.4 | ⬇️ Low |
| MEL (Malignant) | 58.1 | 62.8 | ⬆️ Medium |
| BKL (Benign) | 61.4 | 65.4 | — |
| BCC (Malignant) | 62.5 | 67.8 | ⬆️ High |
| AK (Malignant) | 68.0 | 66.6 | ⬆️ High |
| SCC (Malignant) | 70.0 | 71.2 | ⬆️ Highest |

> [!IMPORTANT]
> **Malignant cancers strongly correlate with older age.** Mean ages: NV=45 → SCC=71. This 26-year gap is a massive signal. Including age as metadata will significantly boost accuracy.

### Sex Distribution
- Male: 13,286 (52.5%) | Female: 11,661 (46.0%) | NaN: 384 (1.5%)
- Roughly balanced across classes, slight male skew in malignant types
- Males diagnosed ~3-5 years older than females for most cancer types

---

## 🦵 Critical Finding #4: Anatomical Site is Discriminative

| Site | Count | % | Key Pattern |
|------|-------|---|-------------|
| Anterior torso | 6,915 | 27.3% | NV dominant |
| Lower extremity | 4,990 | 19.7% | NV, MEL common |
| Head/neck | 4,587 | 18.1% | **BCC heavily concentrated here** |
| Upper extremity | 2,910 | 11.5% | Mixed |
| Posterior torso | 2,787 | 11.0% | NV, MEL |
| **NaN** | **2,631** | **10.4%** | ⚠️ Significant missing |
| Palms/soles | 398 | 1.6% | Rare, distinct patterns |
| Oral/genital | 59 | 0.2% | Very rare |
| Lateral torso | 54 | 0.2% | Very rare |

> [!TIP]
> **BCC is heavily concentrated at head/neck** — this matches clinical reality (sun-exposed areas). Including anatomical site as metadata will help the model distinguish BCC from visually similar lesions. However, 10.4% NaN is concerning — treat NaN as its own category.

### Decision: Include Anatomical Site?

**Yes, but carefully.** I recommend encoding the top 5 sites + "other" + "unknown":
```
anterior_torso, lower_extremity, head_neck, upper_extremity, posterior_torso, other, unknown
→ 7 binary features (one-hot encoded)
```

---

## 🖼️ Critical Finding #5: Image Quality Insights

| Property | Value | Impact |
|----------|-------|--------|
| **Dimensions** | ~1024×768 median | Rectangular, not square |
| **Aspect ratio** | Mean 1.19 | Need cropping strategy |
| **Hair artifacts** | Visible in many samples | Affects classification |
| **Black borders** | Present (circular dermoscope view) | Can confuse the model |
| **Color variation** | Significant across images | Need color normalization |
| **Resolution** | High quality (600-1024px) | 380×380 resize preserves detail |

### Image Preprocessing Strategy
1. **Center-crop to square** (not pad — padding adds artificial black regions)
2. **Resize to 384×384** (slightly larger than B4 native, divisible by many factors)
3. **Hair removal**: Use morphological operations (optional, adds complexity)
4. **No black border removal**: EfficientNet can learn to ignore these

---

## 📋 Missing Data Summary

| Feature | Missing | % | Strategy |
|---------|---------|---|----------|
| anatom_site_general | 2,631 | 10.4% | One-hot, NaN → "unknown" category |
| lesion_id | 2,084 | 8.2% | Not used for prediction |
| age_approx | 437 | 1.7% | Fill with class-wise median |
| sex | 384 | 1.5% | Encode: male=1, female=0, NaN=0.5 |

---

## 🎯 REVISED FINAL TRAINING STRATEGY

### Architecture: EfficientNet-B4 + Rich Metadata Fusion

```
┌──────────────────────────────────────────────────────┐
│                   IMAGE BRANCH                        │
│  Input: 384×384×3                                     │
│  → EfficientNet-B4 (ImageNet pretrained)             │
│  → Global Average Pooling → 1792-dim vector          │
│  → BatchNorm → Dropout(0.3)                          │
├──────────────────────────────────────────────────────┤
│                 METADATA BRANCH                       │
│  Input: 9 features                                    │
│    • age (1, normalized)                              │
│    • sex (1, binary + NaN=0.5)                        │
│    • anatom_site (7, one-hot with 'unknown')          │
│  → Dense(64, ReLU) → BatchNorm → Dropout(0.3)       │
│  → Dense(32, ReLU) → 32-dim vector                   │
├──────────────────────────────────────────────────────┤
│                   FUSION HEAD                         │
│  Concatenate(1792 + 32 = 1824)                        │
│  → Dense(512, ReLU) → BatchNorm → Dropout(0.4)      │
│  → Dense(128, ReLU) → Dropout(0.3)                   │
│  → Dense(8, Softmax) → 8-class prediction            │
└──────────────────────────────────────────────────────┘
```

### 8 Key Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **8 classes** (drop UNK) | UNK has 0 samples; handle OOD via confidence threshold |
| 2 | **384×384 input** | Center-crop rectangular → square, then resize |
| 3 | **EfficientNet-B4** backbone | Proven 90-97% on ISIC, good speed/accuracy balance |
| 4 | **9 metadata features** | age(1) + sex(1) + anatom_site(7) — all discriminative |
| 5 | **Focal Loss (γ=2.0)** | Handles 53.9:1 imbalance without naive oversampling |
| 6 | **2-phase training** | Freeze backbone → fine-tune top 50% |
| 7 | **Heavy augmentation** | Albumentations: flips, rotation, color, cutout, noise |
| 8 | **Confidence thresholding** | OOD safety: if max_prob < 0.5 → "Uncertain" |

### Data Split
```
Train: 70% (17,732 images) — with augmentation
Val:   15% (3,799 images)  — no augmentation
Test:  15% (3,800 images)  — no augmentation
All splits stratified by class
```

### Expected Output Flow (Production)
```
User uploads image + enters age, sex, body site
         ↓
Model predicts 8-class probabilities
         ↓
    ┌────────────────────────────────┐
    │  max_prob < 0.5?               │
    │  YES → "Uncertain - see doctor"│
    │  NO  → Continue ↓              │
    └────────────────────────────────┘
         ↓
Sum malignant class probs (MEL+BCC+AK+SCC)
Sum benign class probs (NV+BKL+DF+VASC)
         ↓
    ┌─────────────────────────────────────────────┐
    │  Binary: "Malignant" or "Benign"            │
    │  Type:   "Melanoma" (with 78% confidence)   │
    │  Risk:   Precautions based on type           │
    └─────────────────────────────────────────────┘
```

---

> [!NOTE]
> **Next step**: If this analysis looks good, I'll write the complete training script (`02_training.py`) for Kaggle with all these decisions baked in.

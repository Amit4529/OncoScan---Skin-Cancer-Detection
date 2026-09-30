"""
============================================================
OncoScan V2 FINAL — Definitive Training Script for Kaggle
============================================================
Lessons from V2 + V2.1:
  V2:   AUC=0.9814 (great), but MEL recall 77%, SCC recall 62% (bad)
  V2.1: MEL recall 84% (great), SCC recall 71%, but AUC=0.9723 (dropped)

Root cause: class weights in loss improved recall but HURT AUC (ranking).

V2 FINAL strategy — get BOTH high AUC AND high recall:
  1. OVERSAMPLING instead of class weights (fix minority exposure without
     distorting loss landscape → preserves AUC)
  2. Label smoothing 0.02 (lighter, better calibration)
  3. 456×456 resolution (proven in V2.1)
  4. TTA 8 transforms (proven +0.8%)
  5. Focal loss γ=2.0, NO class weights (clean gradients → better AUC)
  6. Warmup + cosine LR (proven stable)
  7. Robust save: zip + FileLink + Save Version instructions

Target: Accuracy 90%+, AUC 0.98+, F1 macro 0.88+, MEL recall 85%+

Instructions:
    1. Kaggle notebook → add dataset "andrewmvd/isic-2019"
    2. Enable GPU T4 x2
    3. Cell 1: pip install. Then paste everything else and run.
    4. USE "Save Version" → "Save & Run All" to persist files!
============================================================
"""

# %%==================== CELL 1: SETUP ====================
# !pip install albumentations -q

import os
import json
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateScheduler
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc, f1_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A
import warnings
warnings.filterwarnings('ignore')

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Mixed precision DISABLED — float16 caused NaN with focal loss + multi-GPU
# The gradient computation in focal loss can overflow float16 range (max 65504)
# float32 is ~15% slower but STABLE. Worth the trade-off.
print("ℹ️ Using float32 (mixed precision disabled for stability)")

# Multi-GPU detection
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
    print(f"✅ Multi-GPU: {strategy.num_replicas_in_sync} devices")
else:
    strategy = tf.distribute.get_strategy()
    print(f"✅ Single GPU: {gpus}")

print(f"TensorFlow: {tf.__version__}")

# %%==================== CELL 2: CONFIGURATION ====================
class Config:
    INPUT_DIR = '/kaggle/input'
    OUTPUT_DIR = '/kaggle/working'

    IMG_SIZE = 456
    BATCH_SIZE = 12 * strategy.num_replicas_in_sync  # Scale for multi-GPU
    NUM_CLASSES = 8
    META_FEATURES = 9

    # Phase 1: Frozen backbone
    PHASE1_EPOCHS = 5
    PHASE1_LR = 1e-3

    # Phase 2: Fine-tuning
    PHASE2_EPOCHS = 40
    PHASE2_LR = 1e-4
    MIN_LR = 1e-6
    WARMUP_EPOCHS = 3
    EARLY_STOP_PATIENCE = 12

    # Loss — CLEAN focal loss, NO class weights (oversampling handles balance)
    FOCAL_GAMMA = 2.0
    LABEL_SMOOTHING = 0.02   # Lighter than V2.1's 0.05 → better AUC calibration

    # Oversampling — bring minority classes to 1500+ samples
    MIN_SAMPLES_PER_CLASS = 1500

    META_DROPOUT_RATE = 0.5
    TTA_TRANSFORMS = 8

    CLASS_COLS = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
    CLASS_FULL_NAMES = {
        'MEL': 'Melanoma', 'NV': 'Melanocytic Nevus',
        'BCC': 'Basal Cell Carcinoma', 'AK': 'Actinic Keratosis',
        'BKL': 'Benign Keratosis', 'DF': 'Dermatofibroma',
        'VASC': 'Vascular Lesion', 'SCC': 'Squamous Cell Carcinoma'
    }
    MALIGNANT_INDICES = [0, 2, 3, 7]
    BENIGN_INDICES = [1, 4, 5, 6]
    SITE_CATEGORIES = [
        'anterior_torso', 'head_neck', 'lower_extremity',
        'posterior_torso', 'upper_extremity', 'other', 'unknown'
    ]

cfg = Config()

print(f"\n📐 V2 FINAL Config:")
print(f"   Image:        {cfg.IMG_SIZE}×{cfg.IMG_SIZE}")
print(f"   Batch:        {cfg.BATCH_SIZE} (×{strategy.num_replicas_in_sync} GPU)")
print(f"   Phase 1:      {cfg.PHASE1_EPOCHS} epochs, LR={cfg.PHASE1_LR}")
print(f"   Phase 2:      {cfg.PHASE2_EPOCHS} epochs, LR={cfg.PHASE2_LR}→{cfg.MIN_LR}")
print(f"   Warmup:       {cfg.WARMUP_EPOCHS} epochs")
print(f"   Focal Loss:   γ={cfg.FOCAL_GAMMA}, smoothing={cfg.LABEL_SMOOTHING}")
print(f"   Class weights: NONE (using oversampling instead)")
print(f"   Oversampling: min {cfg.MIN_SAMPLES_PER_CLASS} per class")
print(f"   TTA:          {cfg.TTA_TRANSFORMS} transforms")

# %%==================== CELL 3: DISCOVER DATASET ====================
dataset_dir = None
for root, dirs, files in os.walk(cfg.INPUT_DIR):
    for f in files:
        if 'GroundTruth' in f and f.endswith('.csv'):
            dataset_dir = root
            break
    if dataset_dir:
        break

assert dataset_dir, f"Dataset not found in {cfg.INPUT_DIR}"

gt_csv, meta_csv, image_dir = None, None, None
for f in os.listdir(dataset_dir):
    fpath = os.path.join(dataset_dir, f)
    if 'GroundTruth' in f and f.endswith('.csv'):
        gt_csv = fpath
    elif 'Metadata' in f and f.endswith('.csv'):
        meta_csv = fpath
    elif os.path.isdir(fpath) and 'Training_Input' in f:
        for sr, sd, sf in os.walk(fpath):
            if any(x.endswith('.jpg') for x in sf):
                image_dir = sr
                break

print(f"📁 GT:     {gt_csv}")
print(f"📁 Meta:   {meta_csv}")
print(f"📁 Images: {image_dir} ({len(os.listdir(image_dir)):,} files)")

# %%==================== CELL 4: DATA PREPARATION + OVERSAMPLING ====================
print("Loading CSVs...")
df_gt = pd.read_csv(gt_csv)
df_meta = pd.read_csv(meta_csv)

df_gt['label_idx'] = df_gt[cfg.CLASS_COLS].values.argmax(axis=1)
df_gt['label_name'] = df_gt[cfg.CLASS_COLS].idxmax(axis=1)
df = df_gt[['image', 'label_idx', 'label_name']].merge(df_meta, on='image', how='left')
print(f"Merged: {df.shape[0]:,} samples")

# Image paths
df['image_path'] = df['image'].apply(lambda x: os.path.join(image_dir, x + '.jpg'))
exists_mask = df['image_path'].apply(os.path.exists)
missing = (~exists_mask).sum()
if missing > 0:
    print(f"⚠️ {missing} images not found, dropping")
    df = df[exists_mask].reset_index(drop=True)
print(f"Valid: {df.shape[0]:,} samples")

# ---- METADATA ENCODING ----
df['age_norm'] = df.groupby('label_name')['age_approx'].transform(
    lambda x: x.fillna(x.median())
)
df['age_norm'] = df['age_norm'].fillna(55.0) / 90.0
df['age_norm'] = df['age_norm'].clip(0, 1)

df['sex_enc'] = df['sex'].map({'male': 1.0, 'female': 0.0}).fillna(0.5)

site_map = {
    'anterior torso': 'anterior_torso', 'lower extremity': 'lower_extremity',
    'head/neck': 'head_neck', 'upper extremity': 'upper_extremity',
    'posterior torso': 'posterior_torso', 'lateral torso': 'other',
    'palms/soles': 'other', 'oral/genital': 'other',
}
df['site_clean'] = df['anatom_site_general'].map(site_map).fillna('unknown')
for cat in cfg.SITE_CATEGORIES:
    df[f'site_{cat}'] = (df['site_clean'] == cat).astype(np.float32)

meta_cols = ['age_norm', 'sex_enc'] + [f'site_{c}' for c in cfg.SITE_CATEGORIES]

# ---- STRATIFIED SPLIT 70/15/15 ----
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=SEED, stratify=df['label_idx']
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=SEED, stratify=temp_df['label_idx']
)

print(f"\n📊 Original split:")
print(f"   Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")

# ---- OVERSAMPLING (the key V2 FINAL strategy) ----
print(f"\n🔄 Oversampling minority classes to {cfg.MIN_SAMPLES_PER_CLASS}+ samples:")
oversampled_dfs = []
for class_idx in range(cfg.NUM_CLASSES):
    class_df = train_df[train_df['label_idx'] == class_idx]
    n = len(class_df)
    if n < cfg.MIN_SAMPLES_PER_CLASS:
        oversampled = class_df.sample(
            cfg.MIN_SAMPLES_PER_CLASS, replace=True, random_state=SEED
        )
        print(f"   {cfg.CLASS_COLS[class_idx]:>4s}: {n:>5,} → {cfg.MIN_SAMPLES_PER_CLASS:,} "
              f"(oversampled {cfg.MIN_SAMPLES_PER_CLASS/n:.1f}×)")
    else:
        oversampled = class_df
        print(f"   {cfg.CLASS_COLS[class_idx]:>4s}: {n:>5,} (kept)")
    oversampled_dfs.append(oversampled)

train_df_balanced = pd.concat(oversampled_dfs).sample(
    frac=1, random_state=SEED
).reset_index(drop=True)

print(f"\n   Original training:   {len(train_df):,} samples")
print(f"   Balanced training:  {len(train_df_balanced):,} samples")
print(f"   Increase:           {(len(train_df_balanced) / len(train_df) - 1)*100:.1f}%")

# %%==================== CELL 5: AUGMENTATION ====================
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1, scale_limit=0.15, rotate_limit=30,
        border_mode=0, p=0.6
    ),
    A.OneOf([
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=1
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15, contrast_limit=0.15, p=1
        ),
    ], p=0.6),
    A.OneOf([
        A.GaussNoise(var_limit=(5.0, 30.0), p=1),
        A.GaussianBlur(blur_limit=(3, 5), p=1),
    ], p=0.2),
    A.CoarseDropout(
        max_holes=8, max_height=cfg.IMG_SIZE // 12, max_width=cfg.IMG_SIZE // 12,
        min_holes=2, fill_value=0, p=0.3
    ),
])
print("✅ Augmentation pipeline ready")

# %%==================== CELL 6: tf.data PIPELINES ====================
AUTOTUNE = tf.data.AUTOTUNE

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [cfg.IMG_SIZE, cfg.IMG_SIZE])
    return img

def augment_numpy(image_np):
    image_np = image_np.numpy().astype(np.uint8)
    augmented = train_transform(image=image_np)
    return augmented['image'].astype(np.float32)

def make_dataset(dataframe, is_training=False):
    paths = dataframe['image_path'].values
    metadata = dataframe[meta_cols].values.astype(np.float32)
    labels = tf.one_hot(dataframe['label_idx'].values, depth=cfg.NUM_CLASSES)

    ds = tf.data.Dataset.from_tensor_slices((paths, metadata, labels))
    if is_training:
        ds = ds.shuffle(buffer_size=min(len(dataframe), 20000),
                        seed=SEED, reshuffle_each_iteration=True)

    def process_sample(path, meta, label):
        image = load_image(path)
        if is_training:
            image = tf.py_function(augment_numpy, [image], tf.float32)
            image.set_shape([cfg.IMG_SIZE, cfg.IMG_SIZE, 3])
        return {'image_input': image, 'meta_input': meta}, label

    ds = ds.map(process_sample, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(cfg.BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds

def make_tta_dataset(dataframe, transform_fn):
    paths = dataframe['image_path'].values
    metadata = dataframe[meta_cols].values.astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, metadata))

    def process_sample(path, meta):
        image = load_image(path)
        image = transform_fn(image)
        image = tf.ensure_shape(image, [cfg.IMG_SIZE, cfg.IMG_SIZE, 3])
        return {'image_input': image, 'meta_input': meta}

    ds = ds.map(process_sample, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(cfg.BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds

print("Building pipelines...")
train_ds = make_dataset(train_df_balanced, is_training=True)  # OVERSAMPLED
val_ds = make_dataset(val_df, is_training=False)
test_ds = make_dataset(test_df, is_training=False)

for (batch_x, batch_y) in train_ds.take(1):
    print(f"  Image: {batch_x['image_input'].shape}")
    print(f"  Meta:  {batch_x['meta_input'].shape}")
    print(f"  Label: {batch_y.shape}")
print("✅ Pipelines ready (train uses OVERSAMPLED data)")

# %%==================== CELL 7: CUSTOM COMPONENTS ====================

@tf.keras.utils.register_keras_serializable(package='OncoScan')
class ModalityDropout(layers.Layer):
    """Zero out entire metadata vector with probability `rate` during training."""
    def __init__(self, rate=0.5, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            mask = tf.cast(
                tf.random.uniform([tf.shape(inputs)[0], 1]) > self.rate,
                inputs.dtype
            )
            return inputs * mask
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({'rate': self.rate})
        return config


@tf.keras.utils.register_keras_serializable(package='OncoScan')
class FocalLoss(keras.losses.Loss):
    """Clean Focal Loss with optional label smoothing. NO class weights.

    The key insight from V2→V2.1: class weights in loss hurt AUC.
    Instead, we handle imbalance via OVERSAMPLING in the data pipeline.
    This keeps the loss landscape clean → better probability ranking → higher AUC.
    """
    def __init__(self, gamma=2.0, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)

        if self.label_smoothing > 0:
            y_true = y_true * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1.0 - y_pred, self.gamma)
        loss = focal_weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'label_smoothing': self.label_smoothing,
        })
        return config

print("✅ Custom components: ModalityDropout, FocalLoss (clean, no class weights)")

# %%==================== CELL 8: BUILD MODEL ====================
def build_model():
    image_input = Input(shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 3), name='image_input')
    base_model = EfficientNetB4(
        include_top=False, weights='imagenet',
        input_shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 3), pooling='avg'
    )
    image_features = base_model(image_input)
    image_features = layers.BatchNormalization()(image_features)
    image_features = layers.Dropout(0.3)(image_features)

    meta_input = Input(shape=(cfg.META_FEATURES,), name='meta_input')
    meta = ModalityDropout(rate=cfg.META_DROPOUT_RATE)(meta_input)
    meta = layers.Dense(64, activation='relu')(meta)
    meta = layers.BatchNormalization()(meta)
    meta = layers.Dropout(0.5)(meta)
    meta = layers.Dense(32, activation='relu')(meta)

    combined = layers.Concatenate()([image_features, meta])
    combined = layers.Dense(512, activation='relu')(combined)
    combined = layers.BatchNormalization()(combined)
    combined = layers.Dropout(0.4)(combined)
    combined = layers.Dense(128, activation='relu')(combined)
    combined = layers.Dropout(0.3)(combined)

    output = layers.Dense(
        cfg.NUM_CLASSES, activation='softmax',
        dtype='float32', name='predictions'
    )(combined)

    model = Model(inputs=[image_input, meta_input], outputs=output)
    return model, base_model

# Build inside strategy scope for multi-GPU
with strategy.scope():
    model, backbone = build_model()

print(f"\n📐 Model: EfficientNet-B4 @ {cfg.IMG_SIZE}×{cfg.IMG_SIZE}")
print(f"   Total params: {model.count_params():,}")

# %%==================== CELL 9: PHASE 1 — FROZEN BACKBONE ====================
print("\n" + "=" * 70)
print("🧊 PHASE 1: Feature Extraction (Frozen Backbone)")
print("=" * 70)

backbone.trainable = False

loss_fn = FocalLoss(gamma=cfg.FOCAL_GAMMA, label_smoothing=cfg.LABEL_SMOOTHING)

with strategy.scope():
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.PHASE1_LR, clipnorm=1.0),
        loss=loss_fn,
        metrics=[
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            keras.metrics.AUC(name='auc', multi_label=False),
        ]
    )

trainable_count = sum(
    tf.keras.backend.count_params(w) for w in model.trainable_weights
)
print(f"Trainable params (head only): {trainable_count:,}")
print(f"Epochs: {cfg.PHASE1_EPOCHS}, LR: {cfg.PHASE1_LR}")
print(f"Loss: FocalLoss(γ={cfg.FOCAL_GAMMA}, smoothing={cfg.LABEL_SMOOTHING})")
print(f"NO class weights — oversampled data handles balance")

history_p1 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=cfg.PHASE1_EPOCHS, verbose=1,
    callbacks=[keras.callbacks.TerminateOnNaN()]
)

print(f"\n✅ Phase 1 complete.")
print(f"   Val Accuracy: {history_p1.history['val_accuracy'][-1]:.4f}")
print(f"   Val AUC:      {history_p1.history['val_auc'][-1]:.4f}")

# %%==================== CELL 10: PHASE 2 — FINE-TUNING ====================
print("\n" + "=" * 70)
print("🔥 PHASE 2: Fine-Tuning (Top 50% of Backbone)")
print("=" * 70)

backbone.trainable = True
total_layers = len(backbone.layers)
freeze_until = total_layers // 2
for layer in backbone.layers[:freeze_until]:
    layer.trainable = False

trainable_count = sum(
    tf.keras.backend.count_params(w) for w in model.trainable_weights
)
print(f"Backbone: {total_layers} layers (frozen: {freeze_until}, trainable: {total_layers - freeze_until})")
print(f"Total trainable: {trainable_count:,}")

def warmup_cosine_schedule(epoch):
    if epoch < cfg.WARMUP_EPOCHS:
        lr = cfg.MIN_LR + (cfg.PHASE2_LR - cfg.MIN_LR) * (epoch / cfg.WARMUP_EPOCHS)
    else:
        progress = (epoch - cfg.WARMUP_EPOCHS) / (cfg.PHASE2_EPOCHS - cfg.WARMUP_EPOCHS)
        lr = cfg.MIN_LR + 0.5 * (cfg.PHASE2_LR - cfg.MIN_LR) * (1 + np.cos(np.pi * progress))
    return float(lr)

with strategy.scope():
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=cfg.PHASE2_LR, weight_decay=1e-5, clipnorm=1.0
        ),
        loss=FocalLoss(gamma=cfg.FOCAL_GAMMA, label_smoothing=cfg.LABEL_SMOOTHING),
        metrics=[
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            keras.metrics.AUC(name='auc', multi_label=False),
        ]
    )

callbacks = [
    keras.callbacks.TerminateOnNaN(),
    LearningRateScheduler(warmup_cosine_schedule, verbose=1),
    EarlyStopping(
        monitor='val_auc', patience=cfg.EARLY_STOP_PATIENCE,
        mode='max', restore_best_weights=True, verbose=1
    ),
    ModelCheckpoint(
        os.path.join(cfg.OUTPUT_DIR, 'oncoscan_v2_final_best.keras'),
        monitor='val_auc', save_best_only=True, mode='max', verbose=1
    ),
]

print(f"Epochs: {cfg.PHASE2_EPOCHS}, Warmup: {cfg.WARMUP_EPOCHS}")
print(f"LR: {cfg.PHASE2_LR} → {cfg.MIN_LR} (cosine)")
print(f"EarlyStopping: patience={cfg.EARLY_STOP_PATIENCE} on val_auc")

history_p2 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=cfg.PHASE2_EPOCHS,
    callbacks=callbacks, verbose=1
)

print(f"\n✅ Phase 2 complete.")
print(f"   Best Val Accuracy: {max(history_p2.history['val_accuracy']):.4f}")
print(f"   Best Val AUC:      {max(history_p2.history['val_auc']):.4f}")

# %%==================== CELL 11: TRAINING CURVES ====================
def combine_histories(h1, h2):
    combined = {}
    for key in h1.history:
        combined[key] = h1.history[key] + h2.history[key]
    return combined

full_history = combine_histories(history_p1, history_p2)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for ax, metric, title in zip(
    axes, ['accuracy', 'auc', 'loss'], ['Accuracy', 'AUC', 'Focal Loss']
):
    ax.plot(full_history[metric], label='Train', linewidth=2)
    ax.plot(full_history[f'val_{metric}'], label='Val', linewidth=2)
    ax.axvline(x=cfg.PHASE1_EPOCHS - 0.5, color='gray',
               linestyle='--', alpha=0.5, label='Phase 1→2')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('OncoScan V2 FINAL — Training History', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Training curves saved")

# %%==================== CELL 12: EVALUATION WITH TTA ====================
print("\n" + "=" * 70)
print("📊 COMPREHENSIVE EVALUATION ON TEST SET")
print("=" * 70)

# Standard predictions
print("Running standard predictions...")
y_pred_probs_standard = model.predict(test_ds, verbose=1)

# TTA
print(f"\n🔄 TTA ({cfg.TTA_TRANSFORMS} transforms)...")
tta_transforms = [
    ('Original',    lambda x: x),
    ('H-Flip',      lambda x: tf.image.flip_left_right(x)),
    ('V-Flip',      lambda x: tf.image.flip_up_down(x)),
    ('Rot90',       lambda x: tf.image.rot90(x, k=1)),
    ('Rot180',      lambda x: tf.image.rot90(x, k=2)),
    ('Rot270',      lambda x: tf.image.rot90(x, k=3)),
    ('Rot90+HFlip', lambda x: tf.image.flip_left_right(tf.image.rot90(x, k=1))),
    ('Rot90+VFlip', lambda x: tf.image.flip_up_down(tf.image.rot90(x, k=1))),
]

all_tta_preds = []
for name, tfm_fn in tta_transforms:
    print(f"   {name}...", end=' ')
    tta_ds = make_tta_dataset(test_df, tfm_fn)
    preds = model.predict(tta_ds, verbose=0)
    all_tta_preds.append(preds)
    print(f"✅")

y_pred_probs = np.mean(all_tta_preds, axis=0)
print(f"✅ TTA complete — averaged {len(all_tta_preds)} predictions")

y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = test_df['label_idx'].values
y_true_onehot = np.zeros((len(y_true_classes), cfg.NUM_CLASSES))
y_true_onehot[np.arange(len(y_true_classes)), y_true_classes] = 1

# TTA improvement
std_acc = np.mean(np.argmax(y_pred_probs_standard, axis=1) == y_true_classes)
std_auc = roc_auc_score(y_true_onehot, y_pred_probs_standard, average='macro', multi_class='ovr')

# Overall metrics
test_acc = np.mean(y_pred_classes == y_true_classes)
test_auc_macro = roc_auc_score(y_true_onehot, y_pred_probs, average='macro', multi_class='ovr')
test_auc_weighted = roc_auc_score(y_true_onehot, y_pred_probs, average='weighted', multi_class='ovr')
test_f1_macro = f1_score(y_true_classes, y_pred_classes, average='macro')
test_f1_weighted = f1_score(y_true_classes, y_pred_classes, average='weighted')

print(f"\n📊 TTA Improvement:")
print(f"   Accuracy: {std_acc:.4f} → {test_acc:.4f} ({(test_acc-std_acc)*100:+.2f}%)")
print(f"   AUC:      {std_auc:.4f} → {test_auc_macro:.4f} ({(test_auc_macro-std_auc)*100:+.2f}%)")

print(f"\n{'═' * 55}")
print(f"  OVERALL TEST METRICS (V2 FINAL + TTA)")
print(f"{'═' * 55}")
print(f"  Accuracy:        {test_acc:.4f} ({test_acc*100:.1f}%)")
print(f"  AUC (macro):     {test_auc_macro:.4f}")
print(f"  AUC (weighted):  {test_auc_weighted:.4f}")
print(f"  F1 (macro):      {test_f1_macro:.4f}")
print(f"  F1 (weighted):   {test_f1_weighted:.4f}")
print(f"{'═' * 55}")

# Classification report
print(f"\n📋 Classification Report:")
report = classification_report(
    y_true_classes, y_pred_classes,
    target_names=cfg.CLASS_COLS, digits=4, output_dict=True
)
print(classification_report(
    y_true_classes, y_pred_classes,
    target_names=cfg.CLASS_COLS, digits=4
))

# Binary grouping
malignant_prob = y_pred_probs[:, cfg.MALIGNANT_INDICES].sum(axis=1)
y_pred_binary = (malignant_prob >= 0.5).astype(int)
y_true_binary = np.isin(y_true_classes, cfg.MALIGNANT_INDICES).astype(int)
binary_acc = np.mean(y_pred_binary == y_true_binary)
binary_auc = roc_auc_score(y_true_binary, malignant_prob)

print(f"\n📋 Binary (Malignant vs Benign):")
print(f"  Accuracy: {binary_acc:.4f} ({binary_acc*100:.1f}%)")
print(f"  AUC:      {binary_auc:.4f}")
print(classification_report(
    y_true_binary, y_pred_binary,
    target_names=['Benign', 'Malignant'], digits=4
))

# Confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=cfg.CLASS_COLS, yticklabels=cfg.CLASS_COLS)
axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')

sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
            xticklabels=cfg.CLASS_COLS, yticklabels=cfg.CLASS_COLS)
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')

plt.suptitle('OncoScan V2 FINAL — Confusion Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()

# ROC curves
fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.Set1(np.linspace(0, 1, cfg.NUM_CLASSES))
for i, (cls, color) in enumerate(zip(cfg.CLASS_COLS, colors)):
    fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
    class_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f'{cls} ({cfg.CLASS_FULL_NAMES[cls]}) — AUC: {class_auc:.3f}')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves Per Class (V2 FINAL + TTA)', fontsize=16, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'roc_curves.png'), dpi=150, bbox_inches='tight')
plt.show()

# Confidence distribution
max_probs = np.max(y_pred_probs, axis=1)
correct_mask = y_pred_classes == y_true_classes

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(max_probs[correct_mask], bins=50, alpha=0.6, label='Correct', color='#2ecc71')
ax.hist(max_probs[~correct_mask], bins=50, alpha=0.6, label='Incorrect', color='#e74c3c')
ax.set_xlabel('Max Prediction Confidence', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
ax.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'confidence_dist.png'), dpi=150, bbox_inches='tight')
plt.show()

# Threshold analysis
print(f"\n📊 Confidence Thresholding:")
for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    above = max_probs >= t
    if above.sum() > 0:
        acc = np.mean(y_pred_classes[above] == y_true_classes[above])
        cov = above.mean()
        print(f"  Threshold {t:.1f}: Accuracy={acc:.4f}, Coverage={cov*100:.1f}%")

# V2 / V2.1 / V2 FINAL comparison
print(f"\n{'═' * 65}")
print(f"  FULL VERSION COMPARISON")
print(f"{'═' * 65}")
print(f"  {'Metric':<20} {'V2':>10} {'V2.1':>10} {'V2 FINAL':>10}")
print(f"  {'-'*60}")
print(f"  {'Accuracy':<20} {'0.8716':>10} {'0.8842':>10} {test_acc:>10.4f}")
print(f"  {'AUC (macro)':<20} {'0.9814':>10} {'0.9723':>10} {test_auc_macro:>10.4f}")
print(f"  {'F1 (macro)':<20} {'0.8145':>10} {'0.8550':>10} {test_f1_macro:>10.4f}")
print(f"  {'Binary Acc':<20} {'0.9174':>10} {'0.9211':>10} {binary_acc:>10.4f}")
print(f"  {'Binary AUC':<20} {'0.9670':>10} {'0.9704':>10} {binary_auc:>10.4f}")
print(f"  {'-'*60}")

# %%==================== CELL 13: SAVE & EXPORT (ROBUST) ====================
print("\n" + "=" * 70)
print("💾 SAVING MODEL & RESULTS (ROBUST)")
print("=" * 70)

saved_files = []

# 1. Model save
try:
    p = os.path.join(cfg.OUTPUT_DIR, 'oncoscan_v2_final.keras')
    model.save(p)
    saved_files.append(p)
    print(f"✅ Model: {p}")
except Exception as e:
    print(f"⚠️ Model save failed: {e}")

# 2. Weights save
try:
    p = os.path.join(cfg.OUTPUT_DIR, 'oncoscan_v2_final_weights.weights.h5')
    model.save_weights(p)
    saved_files.append(p)
    print(f"✅ Weights: {p}")
except Exception as e:
    print(f"⚠️ Weights save failed: {e}")

# 3. Config
config_data = {
    'version': 'V2_FINAL',
    'img_size': cfg.IMG_SIZE,
    'num_classes': cfg.NUM_CLASSES,
    'meta_features': cfg.META_FEATURES,
    'class_names': cfg.CLASS_COLS,
    'class_full_names': cfg.CLASS_FULL_NAMES,
    'malignant_indices': cfg.MALIGNANT_INDICES,
    'benign_indices': cfg.BENIGN_INDICES,
    'site_categories': cfg.SITE_CATEGORIES,
    'meta_cols': meta_cols,
    'focal_gamma': cfg.FOCAL_GAMMA,
    'label_smoothing': cfg.LABEL_SMOOTHING,
    'meta_dropout_rate': cfg.META_DROPOUT_RATE,
    'tta_transforms': cfg.TTA_TRANSFORMS,
    'oversampling_min': cfg.MIN_SAMPLES_PER_CLASS,
}
p = os.path.join(cfg.OUTPUT_DIR, 'oncoscan_v2_final_config.json')
with open(p, 'w') as f:
    json.dump(config_data, f, indent=2)
saved_files.append(p)
print(f"✅ Config: {p}")

# 4. Results
results = {
    'config': config_data,
    'metrics': {
        'test_accuracy': float(test_acc),
        'test_auc_macro': float(test_auc_macro),
        'test_auc_weighted': float(test_auc_weighted),
        'test_f1_macro': float(test_f1_macro),
        'test_f1_weighted': float(test_f1_weighted),
        'binary_accuracy': float(binary_acc),
        'binary_auc': float(binary_auc),
    },
    'per_class_report': report,
    'training_history': {
        k: [float(v) for v in vals]
        for k, vals in full_history.items()
    }
}
p = os.path.join(cfg.OUTPUT_DIR, 'oncoscan_v2_final_results.json')
with open(p, 'w') as f:
    json.dump(results, f, indent=2)
saved_files.append(p)
print(f"✅ Results: {p}")

# 5. Create ZIP for easy download
try:
    import zipfile
    zip_path = os.path.join(cfg.OUTPUT_DIR, 'oncoscan_v2_final_package.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in os.listdir(cfg.OUTPUT_DIR):
            fpath = os.path.join(cfg.OUTPUT_DIR, f)
            if os.path.isfile(fpath) and not f.endswith('.zip'):
                zf.write(fpath, f)
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"✅ ZIP package: {zip_path} ({size_mb:.1f} MB)")
except Exception as e:
    print(f"⚠️ ZIP failed: {e}")

# 6. Create download links
try:
    from IPython.display import FileLink, display, HTML
    print("\n📥 Click to download:")
    for f in sorted(os.listdir(cfg.OUTPUT_DIR)):
        fpath = os.path.join(cfg.OUTPUT_DIR, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath) / (1024*1024)
            print(f"   {f} ({size:.1f} MB)")
            display(FileLink(f))
except Exception as e:
    print(f"Download links not available: {e}")

# Final summary
print(f"\n{'═' * 55}")
print(f"  🎉 V2 FINAL TRAINING COMPLETE!")
print(f"{'═' * 55}")
print(f"  Test Accuracy:    {test_acc*100:.1f}%")
print(f"  Test AUC:         {test_auc_macro:.4f}")
print(f"  Test F1 (macro):  {test_f1_macro:.4f}")
print(f"  Binary Accuracy:  {binary_acc*100:.1f}%")
print(f"  Binary AUC:       {binary_auc:.4f}")
print(f"{'═' * 55}")

print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ⚠️ TO KEEP YOUR FILES — DO THIS NOW:                       ║
║                                                              ║
║  1. Click "Save Version" (green button, top right)           ║
║  2. Select "Save & Run All (Commit)"                         ║
║  3. Wait for it to finish                                    ║
║  4. Go to notebook → Versions tab → Latest → Output          ║
║  5. Download from there (files persist forever)               ║
║                                                              ║
║  OR download the ZIP file above RIGHT NOW while              ║
║  the session is still active!                                ║
╚══════════════════════════════════════════════════════════════╝
""")

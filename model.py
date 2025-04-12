# model.py
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def build_model(img_size=224):
    img_input = Input(shape=(img_size, img_size, 3), name='image_input')
    base_model = ResNet50(input_tensor=img_input, include_top=False, weights='imagenet')
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    
    tabular_input = Input(shape=(3,), name='tabular_input')
    y = Dense(16, activation='relu')(tabular_input)
    
    combined = Concatenate()([x, y])
    z = Dense(64, activation='relu')(combined)
    output = Dense(1, activation='sigmoid')(z)
    
    model = Model(inputs=[img_input, tabular_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def train_model(model, train_ds, val_ds, train_labels, epochs=20, model_path='cancer_detection_model.h5'):
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_labels)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True)
    ]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    return history

def evaluate_model(model, test_ds):
    results = model.evaluate(test_ds, return_dict=True)
    print("\nTest Set Performance:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

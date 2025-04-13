# model.py
import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt

def create_model(image_shape=(128, 128, 3), meta_shape=(3,)) -> tf.keras.Model:
    """Builds and compiles a hybrid CNN model with image + metadata input (age, sex, diagnosis_1)."""
    # Image input branch
    image_input = Input(shape=image_shape, name='image_input')
    x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)

    # Metadata branch (age, sex, diagnosis_1)
    meta_input = Input(shape=meta_shape, name='meta_input')
    m = layers.Dense(32, activation='relu')(meta_input)
    m = layers.Dropout(0.3)(m)

    # Combine both branches
    combined = layers.concatenate([x, m])
    combined = layers.Dense(512, activation='relu')(combined)
    combined = layers.Dropout(0.5)(combined)
    combined = layers.Dense(256, activation='relu')(combined)
    combined = layers.Dropout(0.3)(combined)
    output = layers.Dense(1, activation='sigmoid')(combined)

    model = Model(inputs=[image_input, meta_input], outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )

    return model

def train_and_evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test):
    """Trains the model and evaluates on the test set."""
    model = create_model(meta_shape=(3,))  # 3 metadata features: age, sex, diagnosis_1

    # Data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator()

    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"Class weights: {class_weight_dict}")

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint_recall = ModelCheckpoint(
        'best_skin_cancer_recall.h5',
        monitor='val_recall',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    checkpoint_auc = ModelCheckpoint(
        'best_skin_cancer_auc.h5',
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    # Training
    history = model.fit(
        [X_train, meta_train], y_train,
        epochs=30,
        batch_size=32,
        validation_data=([X_val, meta_val], y_val),
        class_weight=class_weight_dict,
        callbacks=[early_stopping, checkpoint_recall, checkpoint_auc, reduce_lr]
    )

    # Evaluation
    test_loss, test_acc, test_auc, test_precision, test_recall = model.evaluate([X_test, meta_test], y_test)
    print(f"Test accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")

    # Classification report
    y_pred = (model.predict([X_test, meta_test]) > 0.5).astype(int)
    print(classification_report(y_test, y_pred))

    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title('Model Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Model Loss')

    plt.show()

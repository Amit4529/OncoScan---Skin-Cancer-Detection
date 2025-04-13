# data_preprocessing.py
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

def load_data(use_cache=True):
    cache_dir = "cached_data"
    os.makedirs(cache_dir, exist_ok=True)

    # Cached file paths
    cached_files = {
        "X_train": os.path.join(cache_dir, "X_train.npy"),
        "X_val": os.path.join(cache_dir, "X_val.npy"),
        "X_test": os.path.join(cache_dir, "X_test.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "y_test": os.path.join(cache_dir, "y_test.npy"),
        "meta_train": os.path.join(cache_dir, "meta_train.npy"),
        "meta_val": os.path.join(cache_dir, "meta_val.npy"),
        "meta_test": os.path.join(cache_dir, "meta_test.npy")
    }

    # Load from cache if available
    if use_cache and all(os.path.exists(f) for f in cached_files.values()):
        print("Loading data from cache...")
        return tuple(np.load(f) for f in cached_files.values())

    # Process from scratch
    print("Processing images from scratch...")
    df = pd.read_csv("bcn20000_cleaned_metadata.csv")
    df = df[df['diagnosis_1'].isin(['Benign', 'Malignant'])]
    df['label'] = df['diagnosis_1'].map({'Benign': 0, 'Malignant': 1})

    image_folder = r"C:\Users\KIIT\Desktop\OncoScan\ISIC-images"
    IMG_SIZE = (128, 128)
    dtype = np.float32

    images, labels = [], []
    ages, sexes, diagnoses = [], [], []
    skipped_files = 0

    for index, row in df.iterrows():
        image_path = os.path.join(image_folder, row['isic_id'] + ".jpg")
        if os.path.exists(image_path):
            try:
                img = cv2.imread(image_path)
                if img is None:
                    skipped_files += 1
                    print(f"Warning: Failed to load image {image_path}")
                    continue
                img = cv2.resize(img, IMG_SIZE)
                img = img / 255.0
                images.append(img.astype(dtype))
                labels.append(row['label'])

                # Metadata: age, sex, and diagnosis_1 encoded as numeric
                ages.append(row['age_approx'] / 100.0)
                sexes.append(1 if row['sex'] == 'Male' else 0)
                diagnoses.append(row['label'])  # 0 for Benign, 1 for Malignant
            except Exception as e:
                skipped_files += 1
                print(f"Error processing {image_path}: {str(e)}")
        else:
            skipped_files += 1
            print(f"Image not found: {image_path}")

    if skipped_files > 0:
        print(f"Total skipped files: {skipped_files}")

    X = np.array(images)
    y = np.array(labels)
    meta = np.array(list(zip(ages, sexes, diagnoses)))  # Now has 3 features

    X_train, X_temp, y_train, y_temp, meta_train, meta_temp = train_test_split(
        X, y, meta, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test, meta_val, meta_test = train_test_split(
        X_temp, y_temp, meta_temp, test_size=0.50, random_state=42, stratify=y_temp)

    # Save cache
    np.save(cached_files["X_train"], X_train)
    np.save(cached_files["X_val"], X_val)
    np.save(cached_files["X_test"], X_test)
    np.save(cached_files["y_train"], y_train)
    np.save(cached_files["y_val"], y_val)
    np.save(cached_files["y_test"], y_test)
    np.save(cached_files["meta_train"], meta_train)
    np.save(cached_files["meta_val"], meta_val)
    np.save(cached_files["meta_test"], meta_test)

    print("Data processing complete. Cached files saved.")
    return X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test

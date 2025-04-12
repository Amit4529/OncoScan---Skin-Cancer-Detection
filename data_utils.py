# data_utils.py
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
BATCH_SIZE = 32
METADATA_PATH = 'HAM10000_metadata.csv'
IMAGE_DIR = 'all_images'  # Use your folder with all images

def load_and_preprocess_data():
    df = pd.read_csv(METADATA_PATH)

    # Binary label: 1 for cancer types, 0 for others
    cancer_types = ['mel', 'bcc', 'akiec']
    df['label'] = df['dx'].apply(lambda x: 1 if x in cancer_types else 0)

    # Fill missing values and one-hot encode sex
    df['age'] = df['age'].fillna(df['age'].median())
    df['sex'] = df['sex'].fillna('unknown')
    df = pd.get_dummies(df, columns=['sex'], drop_first=True)

    # Attach image path using the image_id
    df['image_path'] = df['image_id'].apply(lambda x: os.path.join(IMAGE_DIR, f"{x}.jpg"))

    # Print preview
    print(df[['image_id', 'image_path']].head())

    # Split dataset
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

    print(f"Total samples: {len(df)}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Label distribution (Train):\n{train_df['label'].value_counts(normalize=True)}")

    return train_df, val_df, test_df

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image / 255.0

def process_row(image_path, age, sex_male, sex_unknown, label):
    image = load_image(image_path)
    metadata = tf.stack([age, sex_male, sex_unknown])
    label = tf.cast(label, tf.float32)
    return (image, metadata), label

def df_to_dataset(df, batch_size=BATCH_SIZE, shuffle=True):
    image_paths = df['image_path'].values
    ages = df['age'].values.astype('float32')
    sex_male = df['sex_male'].values.astype('float32')
    sex_unknown = df['sex_unknown'].values.astype('float32')
    labels = df['label'].values.astype('float32')

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, ages, sex_male, sex_unknown, labels)
    )
    dataset = dataset.map(process_row, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
if __name__ == "__main__":
    train_df, val_df, test_df = load_and_preprocess_data()

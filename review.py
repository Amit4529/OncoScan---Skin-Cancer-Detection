# review.py
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import the necessary function from data_utils.py
from data_utils import load_and_preprocess_data, load_image

MODEL_PATH = 'cancer_detection_model.h5'

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load and preprocess data (this will generate test_df)
train_df, val_df, test_df = load_and_preprocess_data()

# Prepare the test data from test_df
X_test_images = np.array([load_image(path) for path in test_df['image_path']])
X_test_images = X_test_images.astype('float32')  # Ensure float32 type for image data

X_test_tabular = test_df[['age', 'sex_male', 'sex_unknown']].values
X_test_tabular = X_test_tabular.astype('float32')  # Ensure float32 type for tabular data

y_test = test_df['label'].values

# Predicting the labels for the test data
y_pred = model.predict([X_test_images, X_test_tabular])
y_pred = np.round(y_pred)  # Convert to binary predictions (0 or 1)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Cancer', 'Cancer'], yticklabels=['No Cancer', 'Cancer'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

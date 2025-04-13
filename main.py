# main.py
from data_preprocessing import load_data
from model import train_and_evaluate_model

# Load data and metadata
X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test = load_data()

# Train and evaluate the model
train_and_evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test)

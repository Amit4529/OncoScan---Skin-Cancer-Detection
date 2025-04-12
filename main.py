# main.py
from data_utils import load_and_preprocess_data, df_to_dataset
from model import build_model, train_model, evaluate_model

def main():
    # Load and preprocess data (binary labels, kabul mein age, gender)
    train_df, val_df, test_df = load_and_preprocess_data()
    
    # Create datasets (images + metadata)
    train_ds = df_to_dataset(train_df, shuffle=True)
    val_ds = df_to_dataset(val_df, shuffle=False)
    test_ds = df_to_dataset(test_df, shuffle=False)
    
    # Build binary classification model
    model = build_model()
    
    # Train model
    train_model(model, train_ds, val_ds, train_df['label'].values)
    
    # Evaluate model
    evaluate_model(model, test_ds)

if __name__ == "__main__":
    main()
# src/train_advanced.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def main():
    """
    Main function to train and evaluate the final, balanced multi-class LightGBM model.
    """
    print("[INFO] Starting final model training pipeline with undersampling...")
    
    # --- 1. Load Data ---
    processed_data_path = 'data/full_processed_dataset.parquet'
    df = pd.read_parquet(processed_data_path)
    print(f"[INFO] Loaded full processed data. Shape: {df.shape}")

    # --- 2. Filter Minority Classes ---
    label_counts = df['Label'].value_counts()
    valid_labels = label_counts[label_counts > 100].index.tolist()
    df = df[df['Label'].isin(valid_labels)]
    print(f"[INFO] Filtered minority classes. Shape: {df.shape}")

    # --- 3. Undersampling the Majority Class (NEW) ---
    print("\n[INFO] Balancing data using undersampling...")
    df_benign = df[df['Label'] == 'Benign']
    df_attacks = df[df['Label'] != 'Benign']

    # We will reduce the number of benign samples to be closer to the number of attack samples.
    # Let's keep 200,000 benign samples, which is close to our largest attack class (DoS).
    df_benign_sampled = df_benign.sample(n=200000, random_state=42)

    # Combine our sampled benign data with all the attack data
    df_balanced = pd.concat([df_benign_sampled, df_attacks])
    print(f"[INFO] Undersampling complete. New balanced dataset shape: {df_balanced.shape}")
    print(f"[INFO] Final balanced class distribution:\n{df_balanced['Label'].value_counts()}")

    # --- 4. Label Encoding ---
    # We now proceed with the new 'df_balanced' DataFrame
    X = df_balanced.drop('Label', axis=1)
    y_str = df_balanced['Label']
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_str)
    
    encoder_path = 'models/label_encoder.joblib'
    joblib.dump(encoder, encoder_path)
    print(f"\n[SUCCESS] Label encoder saved to {encoder_path}")
    
    # --- 5. Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # --- 6. Feature Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- 7. Model Training ---
    # With a more balanced dataset, we may not need the 'class_weight' parameter.
    model = lgb.LGBMClassifier(random_state=42)
    print("\n[INFO] Training final multi-class LightGBM model on balanced data...")
    model.fit(X_train_scaled, y_train)
    print("[SUCCESS] Model training completed.")
    
    # --- 8. Save Artifacts ---
    model_path = 'models/lgbm_multiclass_model.joblib'
    scaler_path = 'models/scaler.joblib'
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"[SUCCESS] Final model saved to {model_path}")
    
    # --- 9. Evaluation ---
    print("\n--- Final Model Evaluation on Balanced Test Set ---")
    y_pred = model.predict(X_train_scaled) # Evaluate on training set first as a sanity check
    target_names = encoder.classes_
    
    print("\n--- Training Set Performance ---")
    print(classification_report(y_train, y_pred, target_names=target_names))

    y_pred_test = model.predict(X_test_scaled)
    print("\n--- Test Set Performance ---")
    print(classification_report(y_test, y_pred_test, target_names=target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Final Confusion Matrix (Undersampled)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    cm_plot_path = 'reports/figures/final_multiclass_confusion_matrix.png'
    plt.savefig(cm_plot_path)
    print(f"\n[SUCCESS] Final confusion matrix plot saved to {cm_plot_path}")

if __name__ == '__main__':
    main()
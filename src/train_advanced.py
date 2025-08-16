# src/train_advanced.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

def main():
    """
    Main function to train and evaluate the advanced LightGBM model.
    """
    print("[INFO] Starting advanced model training pipeline...")
    
    # --- Load Data ---
    processed_data_path = 'data/processed_dataset.parquet'
    df = pd.read_parquet(processed_data_path)
    print(f"[INFO] Loaded processed data. Shape: {df.shape}")
    
    # --- Feature and Target Separation ---
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    # --- Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Data split into training ({len(X_train)} rows) and testing ({len(X_test)} rows).")
    
    # --- Feature Scaling (reusing the same logic) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[INFO] Features scaled using StandardScaler.")
    
    # --- Handle Class Imbalance ---
    # Calculate the ratio of negative to positive samples for the 'scale_pos_weight' parameter
    # This tells the model to pay more attention to the minority class (attacks)
    ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"[INFO] Calculated scale_pos_weight for class imbalance: {ratio:.2f}")

    # --- Model Training ---
    model = lgb.LGBMClassifier(random_state=42, scale_pos_weight=ratio)
    print("[INFO] Training LightGBM model...")
    model.fit(X_train_scaled, y_train)
    print("[SUCCESS] Model training completed.")
    
    # --- Save the Model and Scaler ---
    # We reuse the same scaler, but save the new, better model
    model_path = 'models/lgbm_model.joblib'
    scaler_path = 'models/scaler.joblib' # Overwriting is fine, it's the same scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"[SUCCESS] Advanced model saved to {model_path}")
    print(f"[SUCCESS] Scaler saved to {scaler_path}")
    
    # --- Evaluation ---
    print("\n--- Advanced Model Evaluation on Test Set ---")
    y_pred = model.predict(X_test_scaled)
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
    plt.title('Confusion Matrix (LightGBM)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Save the plot
    cm_plot_path = 'reports/figures/lgbm_confusion_matrix.png'
    plt.savefig(cm_plot_path)
    print(f"\n[SUCCESS] Confusion matrix plot saved to {cm_plot_path}")

if __name__ == '__main__':
    main()
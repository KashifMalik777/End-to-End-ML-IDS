# src/train.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def main():
    """
    Main function to train and evaluate the baseline model.
    """
    print("[INFO] Starting model training pipeline...")
    
    # --- Load Data ---
    processed_data_path = 'data/processed_dataset.parquet'
    df = pd.read_parquet(processed_data_path)
    print(f"[INFO] Loaded processed data. Shape: {df.shape}")
    
    # --- Feature and Target Separation ---
    X = df.drop('Label', axis=1)
    y = df['Label']
    print(f"[INFO] Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")
    
    # --- Train-Test Split ---
    # Stratify by y to ensure the class distribution is the same in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Data split into training ({len(X_train)} rows) and testing ({len(X_test)} rows).")
    
    # --- Feature Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[INFO] Features scaled using StandardScaler.")
    
    # --- Model Training ---
    model = LogisticRegression(random_state=42, max_iter=1000)
    print("[INFO] Training Logistic Regression model...")
    model.fit(X_train_scaled, y_train)
    print("[SUCCESS] Model training completed.")
    
    # --- Save the Model and Scaler ---
    model_path = 'models/model.joblib'
    scaler_path = 'models/scaler.joblib'
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"[SUCCESS] Model saved to {model_path}")
    print(f"[SUCCESS] Scaler saved to {scaler_path}")
    
    # --- Evaluation ---
    print("\n--- Model Evaluation on Test Set ---")
    y_pred = model.predict(X_test_scaled)
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Save the plot
    cm_plot_path = 'reports/figures/confusion_matrix.png'
    plt.savefig(cm_plot_path)
    print(f"\n[SUCCESS] Confusion matrix plot saved to {cm_plot_path}")

if __name__ == '__main__':
    main()
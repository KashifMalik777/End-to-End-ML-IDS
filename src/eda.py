# src/eda.py

import pandas as pd
import numpy as np
import time
from typing import List

# Feature set remains the same
SELECTED_FEATURES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 'Flow Bytes/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
    'Packet Length Std', 'Packet Length Variance', 'Average Packet Size',
    'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
    'Label'
]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the CIC-IDS2017 dataframe by stripping column names,
    handling infinite and NaN values, and removing duplicates.
    """
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs preprocessing and feature engineering on the cleaned dataframe.
    """
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    existing_features = [col for col in SELECTED_FEATURES if col in df.columns]
    df = df[existing_features]
    return df

def main():
    """
    Main function to run the full data processing pipeline on multiple files.
    """
    start_time = time.time()
    
    input_files = [
        'data/monday.csv',
        'data/friday_portscan.csv'
    ]
    output_path = 'data/processed_dataset.parquet'
    
    print("[INFO] Starting data processing pipeline...")
    
    # --- Load, Combine, Clean, and Preprocess ---
    df_list = [pd.read_csv(f) for f in input_files]
    df = pd.concat(df_list, ignore_index=True)
    print(f"Combined initial shape: {df.shape}")
    
    df_clean = clean_data(df.copy())
    print(f"Shape after cleaning: {df_clean.shape}")
    
    df_processed = preprocess_data(df_clean.copy())
    print(f"Shape after preprocessing: {df_processed.shape}")
    
    # --- Verify and Save ---
    print(f"\n[VERIFY] Final check for NaN values: {df_processed.isnull().sum().sum()}")
    if 'Flow Bytes/s' in df_processed.columns:
        # Final check for infinities that can arise from 'Flow Duration' being zero
        df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_processed.dropna(inplace=True)
        print(f"[VERIFY] Final check for infinite values: {df_processed.isin([np.inf, -np.inf]).sum().sum()}")
        print(f"Shape after final infinity check: {df_processed.shape}")

    print(f"\nNew label distribution:\n{df_processed['Label'].value_counts(normalize=True)}")
    
    df_processed.to_parquet(output_path)
    print(f"\n[SUCCESS] Processed data saved to {output_path}")
    
    end_time = time.time()
    print(f"\nPipeline completed in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
# src/eda.py

import pandas as pd
import numpy as np
import time
from typing import List
import os

# --- Configuration ---
# List of all data files we will now process
INPUT_FILES = [
    'data/monday.csv',
    'data/friday_portscan.csv',
    'data/wednesday.csv',
    'data/thursday_morning_webattacks.csv',
    'data/thursday_afternoon_infiltration.csv'
]

# Feature set remains the same for now
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
    'Label' # We'll keep the Label column for now and process it
]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a dataframe by stripping column names, handling infinite/NaN, and removing duplicates.
    """
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs preprocessing: consolidates labels and selects final features.
    """
    # --- 1. Consolidate Attack Labels (NEW) ---
    # Many attack labels exist. We'll group them into broader categories.
    df['Label'] = df['Label'].astype(str) # Ensure Label column is string type
    
    # Define mappings from specific labels to broader categories
    # Using .str.contains() for flexible matching
    conditions = [
        df['Label'].str.contains('BENIGN', case=False, na=False),
        df['Label'].str.contains('PortScan', case=False, na=False),
        df['Label'].str.contains('DoS', case=False, na=False),
        df['Label'].str.contains('DDoS', case=False, na=False),
        df['Label'].str.contains('Web Attack', case=False, na=False),
        df['Label'].str.contains('Infiltration', case=False, na=False),
        df['Label'].str.contains('Bot', case=False, na=False)
    ]
    categories = [
        'Benign', 'PortScan', 'DoS', 'DDoS', 'Web Attack', 'Infiltration', 'Botnet'
    ]
    
    # np.select is an efficient way to apply these conditions
    df['Label'] = np.select(conditions, categories, default='Other')
    
    # --- 2. Select Final Features ---
    existing_features = [col for col in SELECTED_FEATURES if col in df.columns]
    df = df[existing_features]
    
    return df

def main():
    """
    Main function to run the full data processing pipeline on all files.
    """
    start_time = time.time()
    
    output_path = 'data/full_processed_dataset.parquet'
    
    print("[INFO] Starting full data processing pipeline...")
    
    # --- Load, Combine, Clean, and Preprocess ---
    df_list = []
    for f in INPUT_FILES:
        if os.path.exists(f):
            print(f"[INFO] Loading {f}...")
            df_list.append(pd.read_csv(f, encoding='cp1252')) # Added encoding for compatibility
        else:
            print(f"[WARN] File not found, skipping: {f}")

    if not df_list:
        print("[ERROR] No data files found. Exiting.")
        return

    full_df = pd.concat(df_list, ignore_index=True)
    print(f"\nCombined initial shape: {full_df.shape}")
    
    df_clean = clean_data(full_df.copy())
    print(f"Shape after cleaning: {df_clean.shape}")
    
    df_processed = preprocess_data(df_clean.copy())
    print(f"Shape after preprocessing: {df_processed.shape}")
    
    # --- Final Verification and Save ---
    # This check for infinities in 'Flow Bytes/s' is still relevant
    if 'Flow Bytes/s' in df_processed.columns:
        df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_processed.dropna(inplace=True)
        print(f"Shape after final infinity check: {df_processed.shape}")

    print(f"\n[SUCCESS] New multi-class label distribution:\n{df_processed['Label'].value_counts()}")
    
    df_processed.to_parquet(output_path)
    print(f"\n[SUCCESS] Full processed data saved to {output_path}")
    
    end_time = time.time()
    print(f"\nPipeline completed in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
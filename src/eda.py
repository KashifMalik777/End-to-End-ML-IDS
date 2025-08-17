# src/eda.py

import pandas as pd
import numpy as np
import time
import os
from typing import List, Set

# --- Configuration ---
# Add our new 2018 file to the list
INPUT_FILES = [
    'data/monday.csv',
    'data/friday_portscan.csv',
    'data/wednesday.csv',
    'data/thursday_morning_webattacks.csv',
    'data/thursday_afternoon_infiltration.csv',
    'data/bruteforce_2018.csv' # <-- New 2018 data
]

# This list now defines our 'ideal' feature set. The script will find which of these exist in all files.
CORE_FEATURES = [
    'flow_duration', 'total_fwd_packets', 'total_backward_packets',
    'total_length_of_fwd_packets', 'total_length_of_bwd_packets',
    'fwd_packet_length_mean', 'bwd_packet_length_mean', 'flow_bytes_s',
    'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min',
    'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max', 'fwd_iat_min',
    'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min',
    'min_packet_length', 'max_packet_length', 'packet_length_mean',
    'packet_length_std', 'packet_length_variance', 'average_packet_size',
    'avg_fwd_segment_size', 'avg_bwd_segment_size',
    'label'
]

# --- Helper Functions ---

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names to a consistent format: lowercase, underscore-separated.
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_')
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a dataframe by handling infinite/NaN values and removing duplicates.
    """
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs preprocessing: consolidates labels into a common format.
    """
    df['label'] = df['label'].astype(str)
    
    # Expanded conditions to include new attack types from the 2018 dataset
    conditions = [
        df['label'].str.contains('BENIGN', case=False, na=False),
        df['label'].str.contains('PortScan', case=False, na=False),
        df['label'].str.contains('DoS', case=False, na=False),
        df['label'].str.contains('DDoS', case=False, na=False),
        df['label'].str.contains('Web Attack', case=False, na=False),
        df['label'].str.contains('Infiltration', case=False, na=False),
        df['label'].str.contains('Bot', case=False, na=False),
        # New conditions for 2018 data (FTP-Patator, SSH-Patator are Brute Force)
        df['label'].str.contains('BruteForce', case=False, na=False),
        df['label'].str.contains('Patator', case=False, na=False)
    ]
    categories = [
        'Benign', 'PortScan', 'DoS', 'DDoS', 'Web Attack', 'Infiltration', 'Botnet', 'BruteForce', 'BruteForce'
    ]
    
    df['label'] = np.select(conditions, categories, default='Other')
    return df

# --- Main Pipeline ---

def main():
    start_time = time.time()
    output_path = 'data/unified_processed_dataset.parquet'
    
    print("[INFO] Starting universal data processing pipeline...")
    
    # --- 1. Load and Normalize all DataFrames ---
    dataframes = []
    available_columns: List[Set[str]] = []
    
    for f in INPUT_FILES:
        if os.path.exists(f):
            print(f"[INFO] Loading {f}...")
            try:
                df = pd.read_csv(f, encoding='cp1252')
                df = normalize_columns(df)
                dataframes.append(df)
                available_columns.append(set(df.columns))
            except Exception as e:
                print(f"[ERROR] Could not process file {f}: {e}")
        else:
            print(f"[WARN] File not found, skipping: {f}")

    if not dataframes:
        print("[ERROR] No data files loaded. Exiting.")
        return

    # --- 2. Find Common Features ---
    # Find the intersection of column names across all loaded files
    common_columns = list(set.intersection(*available_columns))
    print(f"\n[INFO] Found {len(common_columns)} common columns across all datasets.")

    # We will only use the intersection of our 'ideal' feature list and the common columns
    final_features = [col for col in CORE_FEATURES if col in common_columns]
    print(f"[INFO] Using {len(final_features)} final features for the model.")

    # --- 3. Unify and Process Data ---
    # Select only the final features from each dataframe and concatenate
    unified_df_list = [df[final_features] for df in dataframes]
    unified_df = pd.concat(unified_df_list, ignore_index=True)
    
    print(f"\nCombined initial shape: {unified_df.shape}")

    df_clean = clean_data(unified_df.copy())
    print(f"Shape after cleaning: {df_clean.shape}")
    
    df_processed = preprocess_data(df_clean.copy())
    print(f"Shape after preprocessing: {df_processed.shape}")
    
    if 'flow_bytes_s' in df_processed.columns:
        df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_processed.dropna(inplace=True)
        print(f"Shape after final infinity check: {df_processed.shape}")

    print(f"\n[SUCCESS] New unified label distribution:\n{df_processed['label'].value_counts()}")
    
    df_processed.to_parquet(output_path)
    print(f"\n[SUCCESS] Unified processed data saved to {output_path}")
    
    end_time = time.time()
    print(f"\nPipeline completed in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
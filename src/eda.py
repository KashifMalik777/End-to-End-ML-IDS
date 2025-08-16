# src/eda.py

import pandas as pd
import numpy as np
import time

# Set a random seed for reproducibility
np.random.seed(42)

def main():
    """
    Main function to run the EDA and cleaning process.
    """
    start_time = time.time()
    
    # Define the path to the data
    file_path = 'data/monday.csv'
    
    print("--- Section 1: Initial Data Loading & Inspection ---")
    
    # --- Smoke Test: Load a small sample first ---
    print("\n[INFO] Performing smoke test by loading first 200 rows...")
    try:
        sample_df = pd.read_csv(file_path, nrows=200)
        print("Smoke test successful. Sample data loaded:")
        print(sample_df.head(3))
    except Exception as e:
        print(f"[ERROR] Smoke test failed: {e}")
        return

    # --- Full Data Load ---
    print("\n[INFO] Loading the full dataset. This might take a moment...")
    df = pd.read_csv(file_path)
    print(f"Full dataset loaded successfully. Shape: {df.shape}")
    
    # --- Initial Inspection ---
    print(f"Initial memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\n--- Section 2: Data Cleaning ---")
    
    # --- 1. Clean Column Names ---
    # The dataset has leading spaces in column names.
    original_columns = df.columns.tolist()
    df.columns = df.columns.str.strip()
    print("\n[CLEAN] Stripped leading/trailing whitespace from column names.")
    
    # --- 2. Handle Infinite Values ---
    # Replace infinite values with NaN, as they are problematic for many ML algorithms.
    inf_counts = df.isin([np.inf, -np.inf]).sum().sum()
    print(f"\n[INFO] Found {inf_counts} infinite values.")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("[CLEAN] Replaced infinite values with NaN.")

    # --- 3. Handle Missing Values (NaN) ---
    # Now we drop all rows that contain any NaN values (this includes the ones we just converted from inf).
    nan_counts = df.isnull().sum().sum()
    print(f"\n[INFO] Found {nan_counts} total NaN values (including converted infinities).")
    initial_rows = len(df)
    df.dropna(inplace=True)
    rows_after_dropna = len(df)
    print(f"[CLEAN] Dropped {initial_rows - rows_after_dropna} rows containing NaN values.")
    print(f"Shape after dropping NaNs: {df.shape}")

    # --- 4. Handle Duplicate Rows ---
    duplicate_counts = df.duplicated().sum()
    print(f"\n[INFO] Found {duplicate_counts} duplicate rows.")
    if duplicate_counts > 0:
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        rows_after_drop_duplicates = len(df)
        print(f"[CLEAN] Dropped {initial_rows - rows_after_drop_duplicates} duplicate rows.")
        print(f"Shape after dropping duplicates: {df.shape}")
    
    print("\n--- Section 3: Final Verification ---")
    final_nan = df.isnull().sum().sum()
    final_inf = df.isin([np.inf, -np.inf]).sum().sum()
    print(f"\n[VERIFY] Final check for NaN values: {final_nan}")
    print(f"[VERIFY] Final check for infinite values: {final_inf}")
    print(f"[VERIFY] Final dataset shape: {df.shape}")
    print(f"Final memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    end_time = time.time()
    print(f"\nEDA and cleaning completed in {end_time - start_time:.2f} seconds.")


if __name__ == '__main__':
    main()
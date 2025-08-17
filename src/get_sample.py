# src/get_sample.py
import pandas as pd
import json

def main():
    """
    Selects a sample row from the attack data and formats it for a curl command.
    """
    # Load the raw attack data
    df = pd.read_csv('data/friday_portscan.csv')
    
    # --- THE FIX IS HERE: Clean the column names just like in our EDA script ---
    df.columns = df.columns.str.strip()
    
    # Select a sample row that the model has not been trained on
    # Let's pick row 150000 as our sample
    attack_sample = df.iloc[[150000]]
    
    # Get the feature names our model expects (from train_advanced.py)
    expected_features = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
        'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 'Flow Bytes/s',
        'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
        'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
        'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
        'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
        'Packet Length Std', 'Packet Length Variance', 'Average Packet Size',
        'Avg Fwd Segment Size', 'Avg Bwd Segment Size'
    ]
    
    # Create a dictionary with just those features
    sample_dict = attack_sample[expected_features].to_dict(orient='records')[0]
    
    # Print the JSON payload for the curl command
    print("\n--- Copy the JSON payload below for your curl command ---")
    print(json.dumps(sample_dict))
    print("\n")

if __name__ == "__main__":
    main()
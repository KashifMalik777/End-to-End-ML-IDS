# src/server.py

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# --- 1. Initialize FastAPI App ---
app = FastAPI(
    title="ML IDS API",
    description="API for predicting network intrusion using a LightGBM model.",
    version="1.0.0"
)

# --- 2. Load Model and Scaler ---
try:
    model = joblib.load('models/lgbm_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    print("[INFO] Model and scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"[ERROR] Failed to load model/scaler: {e}")
    model = None
    scaler = None

# --- 3. Define the Input Data Schema ---
class NetworkFeatures(BaseModel):
    Flow_Duration: float = Field(alias='Flow Duration')
    Total_Fwd_Packets: float = Field(alias='Total Fwd Packets')
    Total_Backward_Packets: float = Field(alias='Total Backward Packets')
    Total_Length_of_Fwd_Packets: float = Field(alias='Total Length of Fwd Packets')
    Total_Length_of_Bwd_Packets: float = Field(alias='Total Length of Bwd Packets')
    Fwd_Packet_Length_Mean: float = Field(alias='Fwd Packet Length Mean')
    Bwd_Packet_Length_Mean: float = Field(alias='Bwd Packet Length Mean')
    Flow_Bytes_s: float = Field(alias='Flow Bytes/s')
    Flow_IAT_Mean: float = Field(alias='Flow IAT Mean')
    Flow_IAT_Std: float = Field(alias='Flow IAT Std')
    Flow_IAT_Max: float = Field(alias='Flow IAT Max')
    Flow_IAT_Min: float = Field(alias='Flow IAT Min')
    Fwd_IAT_Mean: float = Field(alias='Fwd IAT Mean')
    Fwd_IAT_Std: float = Field(alias='Fwd IAT Std')
    Fwd_IAT_Max: float = Field(alias='Fwd IAT Max')
    Fwd_IAT_Min: float = Field(alias='Fwd IAT Min')
    Bwd_IAT_Mean: float = Field(alias='Bwd IAT Mean')
    Bwd_IAT_Std: float = Field(alias='Bwd IAT Std')
    Bwd_IAT_Max: float = Field(alias='Bwd IAT Max')
    Bwd_IAT_Min: float = Field(alias='Bwd IAT Min')
    Min_Packet_Length: float = Field(alias='Min Packet Length')
    Max_Packet_Length: float = Field(alias='Max Packet Length')
    Packet_Length_Mean: float = Field(alias='Packet Length Mean')
    Packet_Length_Std: float = Field(alias='Packet Length Std')
    Packet_Length_Variance: float = Field(alias='Packet Length Variance')
    Average_Packet_Size: float = Field(alias='Average Packet Size')
    Avg_Fwd_Segment_Size: float = Field(alias='Avg Fwd Segment Size')
    Avg_Bwd_Segment_Size: float = Field(alias='Avg Bwd Segment Size')

    class Config:
        populate_by_name = True

# --- 4. Create the Prediction Endpoint (Corrected) ---
@app.post("/predict")
def predict(features: NetworkFeatures):
    if not model or not scaler:
        return {"error": "Model not loaded. Please check server logs."}

    # Use .model_dump(by_alias=True) to get a dict with the original column names (with spaces, slashes, etc.)
    feature_dict = features.model_dump(by_alias=True)
    
    # Convert to a DataFrame
    df = pd.DataFrame([feature_dict])

    # Get the feature names the scaler expects, IN THE CORRECT ORDER
    expected_features = scaler.get_feature_names_out()
    
    # Reorder the DataFrame's columns to match the scaler's expected order
    df = df[expected_features]

    # Scale the features
    scaled_features = scaler.transform(df)
    
    # Make a prediction
    prediction = model.predict(scaled_features)
    prediction_label = "Benign" if prediction[0] == 0 else "Attack"
    
    return {"prediction": int(prediction[0]), "label": prediction_label}

@app.get("/health")
def health_check():
    return {"status": "ok"}
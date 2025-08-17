# ML-IDS: A Production-Grade Intrusion Detection System

This project is an end-to-end machine learning application that detects network intrusions. It uses the CIC-IDS2017 dataset to train a LightGBM model, which is then served via a high-performance FastAPI application and containerized with Docker.

## Project Structure

```
ml-ids/
├── data/                 # Holds raw and processed data
├── models/               # Holds trained model and scaler artifacts
├── reports/              # Holds evaluation reports (e.g., confusion matrix)
├── src/                  # All Python source code
├── tests/                # Unit tests for the project
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── pytest.ini
└── requirements.txt
```

## How to Run

### 1. Local Setup (Without Docker)

**Prerequisites:** Python 3.11

```bash
# 1. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the data processing pipeline
python src/eda.py

# 4. Run the model training pipeline
python src/train_advanced.py

# 5. Run the API server
uvicorn src.server:app --reload
```

### 2. Running with Docker (Recommended)

**Prerequisites:** Docker Desktop

```bash
# 1. Build and run the container in the background
docker-compose up --build -d

# 2. Test the API
# The server will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000)
curl -X POST "[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)" -H "Content-Type: application/json" -d "{\"Flow Duration\": 27, \"Total Fwd Packets\": 1, \"Total Backward Packets\": 1, \"Total Length of Fwd Packets\": 0, \"Total Length of Bwd Packets\": 6, \"Fwd Packet Length Mean\": 0.0, \"Bwd Packet Length Mean\": 6.0, \"Flow Bytes/s\": 222222.2222, \"Flow IAT Mean\": 27.0, \"Flow IAT Std\": 0.0, \"Flow IAT Max\": 27, \"Flow IAT Min\": 27, \"Fwd IAT Mean\": 0.0, \"Fwd IAT Std\": 0.0, \"Fwd IAT Max\": 0, \"Fwd IAT Min\": 0, \"Bwd IAT Mean\": 0.0, \"Bwd IAT Std\": 0.0, \"Bwd IAT Max\": 0, \"Bwd IAT Min\": 0, \"Min Packet Length\": 0, \"Max Packet Length\": 6, \"Packet Length Mean\": 2.0, \"Packet Length Std\": 3.464101615, \"Packet Length Variance\": 12.0, \"Average Packet Size\": 3.0, \"Avg Fwd Segment Size\": 0.0, \"Avg Bwd Segment Size\": 6.0}"

# 3. Stop the container
docker-compose down
```
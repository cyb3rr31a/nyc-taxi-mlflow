# NYC Taxi Fare Predictor — ML Pipeline with MLflow

An end-to-end machine learning pipeline that trains models to predict NYC taxi fares, tracks experiments with MLflow, registers the best model, and serves predictions via a FastAPI REST endpoint.

## What this project does

- **Prepares** cleaned NYC taxi trip data for ML training
- **Trains** three models and tracks every experiment automatically with MLflow
- **Compares** runs side by side in the MLflow UI to select the best model
- **Registers** the winning model in the MLflow Model Registry
- **Serves** predictions via a production-style FastAPI endpoint

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 1.6868 | 19.1799 | 0.9205 |
| Random Forest | 0.9061 | 9.4178 | 0.9609 |
| **Gradient Boosting** | **0.8905** | **9.2894** | **0.9615** |

Gradient Boosting won with an R² of 0.96 — meaning the model explains 96% of the variance in fare amounts.

## Features used

| Feature | Description |
|---|---|
| `trip_distance` | Distance of the trip in miles |
| `passenger_count` | Number of passengers |
| `PULocationID` | NYC taxi zone where trip started |
| `DOLocationID` | NYC taxi zone where trip ended |
| `payment_type` | Payment method used |
| `trip_duration_minutes` | Duration of the trip in minutes |

## API

Once running, the API exposes three endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API status |
| `/health` | GET | Health check |
| `/predict` | POST | Predict fare from trip details |

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "trip_distance": 3.5,
    "passenger_count": 1,
    "PULocationID": 162,
    "DOLoactionID": 237,
    "payment_type": 1,
    "trip_duration_minutes": 12.5
  }'
```

**Example response:**

```json
{"predicted_fare": 16.96, "model_version": "1"}
```

Interactive API docs available at `http://localhost:8000/docs`.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/cyb3rr31a/nyc-taxi-mlflow.git
cd nyc-taxi-mlflow
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/Scripts/activate  # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the data

Go to [nyc.gov/site/tlc/about/tlc-trip-record-data.page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) and download the Yellow Taxi parquet file for January 2025. Place it in the project root as `yellow_tripdata_cleaned.parquet`.

### 5. Start the MLflow tracking server

In a separate terminal:

```bash
.venv\Scripts\activate
mlflow ui
```

Opens at `http://localhost:5000`.

### 6. Prepare the data

```bash
python prepare_data.py
```

### 7. Train the models

```bash
python train.py
```

All three models will train and log to MLflow automatically. Visit `http://localhost:5000` to compare runs.

### 8. Register the best model

```bash
python register_model.py
```

### 9. Start the API

```bash
uvicorn api:app --reload
```

Opens at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the interactive Swagger UI.

## Project structure

```
nyc-taxi-mlflow/
├── prepare_data.py          # Data loading and feature preparation
├── train.py                 # Model training and MLflow experiment tracking
├── register_model.py        # Registers the best model in MLflow Model Registry
├── api.py                   # FastAPI endpoint serving predictions
├── mlruns/                  # MLflow experiment data (not tracked by Git)
├── .gitignore               # Ignores mlruns, venv, parquet files
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Dependencies

- `mlflow` — experiment tracking and model registry
- `scikit-learn` — model training
- `fastapi` — REST API
- `uvicorn` — ASGI server
- `pandas` — data manipulation
- `pyarrow` — reading Parquet files

## How this fits into the broader stack

```
NYC Taxi Data (Parquet)
    └── prepare_data.py (feature engineering)
            └── train.py (MLflow experiment tracking)
                    └── register_model.py (Model Registry)
                            └── api.py (FastAPI serving)
```
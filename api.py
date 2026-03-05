from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd

app = FastAPI(
    title="NYC Taxi Fare Predictor",
    description="Predicts taxi fare based on trip data",
    version="1.0"
)

# Load the registered model
print("Loading model...")
model = mlflow.sklearn.load_model("models:/nyc-taxi-fare-predictor/1")
print("Model loaded successfully")

class TripFeatures(BaseModel):
    trip_distance: float
    passenger_count: int
    PULocationID: int
    DOLocationID: int
    payment_type: int
    trip_duration_minutes: float

class FarePrediction(BaseModel):
    predicted_fare: float
    model_version: str = "1"

@app.get("/")
def root():
    return {"messgae": "NYC Taxi Fare Prediction API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=FarePrediction)
def predict(trip: TripFeatures):
    features = pd.DataFrame([trip.model_dump()])
    prediction = model.predict(features)[0]
    return FarePrediction(predicted_fare=round(float(prediction), 2))

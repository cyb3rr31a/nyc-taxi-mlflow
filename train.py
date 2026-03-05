import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_prepare(parquet_path):
    df = pd.read_parquet(parquet_path)
    features = [
        "trip_distance",
        "passenger_count",
        "PULocationID",
        "DOLocationID",
        "payment_type",
        "trip_duration_minutes"
    ]

    target = "fare_amount"
    df = df[features + [target]].dropna()
    df = df[df[target] > 0]
    df = df[df[target] <= 100]
    return df[features], df[target]

def evaluate(y_true, y_pred):
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(mean_squared_error(y_true, y_pred), 4),
        "r2": round(r2_score(y_true, y_pred), 4)
    }

def train_and_log(name, model, params, X_train, X_test, y_train, y_test):
    print(f"Training {name}...")
    
    with mlflow.start_run(run_name=name):
        # Log parameters
        mlflow.log_params(params)

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate
        metrics = evaluate(y_test, y_pred)
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"    MAE:    {metrics['mae']}")
        print(f"    RMSE:   {metrics['rmse']}")
        print(f"    R²:     {metrics['r2']}")

    return metrics

# --- Main ---
mlflow.set_experiment("nyc-taxi-fare-prediction")

print("Loading data...")
X, y = load_and_prepare("yellow_taxi_tripdata.parquet")

# Sample 200k rows for faster training
X = X.sample(200_000, random_state=42)
y = y.loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train size: {len(X_train):,}")
print(f"Test size:  {len(X_test):,}")

# --- Models to train ---
models = [
    (
        "Linear Regression", LinearRegression(), {"model_type": "linear_regression"}
    ),
    (
        "Random Forest", RandomForestRegressor(n_estimators=100, max_depth=10,random_state=42, n_jobs=1), 
        {"model_type": "random_forest", "n_estimators": 100, "max_depth": 10}
    ),
    (
        "Gradient Boosting", GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42), 
        {"model_type": "gradient_boosting", "n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}
    )
]

# --- Train and log each model ---
results = {}
for name, model, params in models:
    results[name] = train_and_log(name, model, params, X_train, X_test, y_train, y_test)

# --- Summary ---
print("\n=== Results Summary ===")
print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("-" * 55)
for name, metrics in results.items():
    print(f"{name:<25} {metrics['mae']:>8} {metrics['rmse']:>8} {metrics['r2']:>8}")

best = min(results, key=lambda x: results[x]['mae'])
print(f"\nBest model: {best} (lowest MAE)")
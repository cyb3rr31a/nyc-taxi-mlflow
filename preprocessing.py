import pandas as pd

def load_and_prepare(parquet_path):
    print("Loading data...")
    df = pd.read_parquet(parquet_path)
    print(f"Rows loaded: {len(df):,}")
    # print column names
    print(f"Columns: {df.columns.tolist()}")

    # Select features and target
    features = [
        "trip_distance",
        "passenger_count",
        "PULocationID",
        "DOLocationID",
        "payment_type",
        "trip_duration_minutes"
    ]
    target = "fare_amount"

    # Drop rows with nulls in selected columns
    df = df[features + [target]].dropna()

    # Remove outliers in target
    df = df[df[target] > 0]
    df = df[df[target] <= 100]

    print(f"Rows after preparation: {len(df):,}")
    print(f"\nFeatures: {features}")
    print(f"Target: {target}")
    print(f"\nSample:\n{df.head()}")

    return df[features], df[target]

X, y = load_and_prepare("yellow_taxi_tripdata.parquet")
print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"\nTarget stats:\n{y.describe()}")

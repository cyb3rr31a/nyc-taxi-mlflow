import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get the best run from the experiment
experiment = client.get_experiment_by_name("nyc-taxi-fare-prediction")
if experiment is None:
    raise ValueError("Experiment 'nyc-taxi-fare-prediction' not found")
runs = client.search_runs(
    experiment_ids = [experiment.experiment_id],
    order_by = ["metrics.mae ASC"],
    max_results = 1
)

best_run = runs[0]
print(f"Best run: {best_run.info.run_id}")
print(f"MAE: {best_run.data.metrics['mae']}")
print(f"Model: {best_run.data.params['model_type']}")

# Register the model
model_uri = f"runs:/{best_run.info.run_id}/model"
mlflow.register_model(model_uri, "nyc-taxi-fare-predictor")
print("\nModel registered successfully")
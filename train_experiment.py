import mlflow

mlflow.set_experiment("my-first-experiment")

with mlflow.start_run():
    mlflow.log_param("model_type", "dummy_model")
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("loss", 0.15)
    print("Experiment logged successfully!")
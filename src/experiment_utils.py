import logging
import os

import mlflow
from mlflow.models import infer_signature

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_mlflow_tracking():
    """
    Set up MLflow tracking server credentials and URI.
    """
    uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(uri)
    logger.info("MLflow tracking URI and credentials set.")

    return mlflow


import mlflow

def log_model_to_mlflow(
    model,
    input_data,
    experiment_name,
    metric_name="metric",
    model_name=None,
    params=None,
    score=None,
):
    """
    Log a trained model, parameters, and metrics to MLflow and print log links.
    """
    try:
        # Set MLflow experiment
        mlflow.set_experiment(experiment_name)

        # Start MLflow run
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id

            # Log parameters
            if params:
                mlflow.log_params(params)

            # Log metric
            if score is not None:
                mlflow.log_metric(metric_name, score)

            # Infer the model signature
            signature = mlflow.models.infer_signature(input_data, model.predict(input_data))

            # Assign model name if not provided
            if not model_name:
                model_name = model.__class__.__name__

            # Log model
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model_artifact",
                signature=signature,
                input_example=input_data,
                registered_model_name=model_name,
            )

            # Construct MLflow Log Links
            tracking_uri = mlflow.get_tracking_uri()
            run_link = f"{tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}"
            experiment_link = f"{tracking_uri}/#/experiments/{experiment_id}"

            print(f"🔗 View run at: {run_link}")
            print(f"📂 View experiment at: {experiment_link}")

            return model_info

    except Exception as e:
        print(f"❌ Error logging to MLflow: {e}")
        raise
from contextlib import contextmanager
from typing import Dict, Any

import mlflow
from .config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI


def init_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


@contextmanager
def mlflow_run(run_name: str, params: Dict[str, Any]):
    """Context manager that initialises an MLflow run with given params."""
    init_mlflow()
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        yield

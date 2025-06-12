import os

import mlflow
import pandas as pd

from src import logger


class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.artifacts_root = self.config.artifacts_root
        self.prediction_file_path = self.config.prediction_file_path
        self.mlflow_run_id = self.config.mlflow_run_id
        self.predictions_df = None

        if not self.mlflow_run_id and not self.prediction_file_path:
            logger.error(
                "Either 'mlflow_run_id' or 'prediction_file_path' must be provided in the config."
            )

    def _get_predictions_from_mlflow(self):
        if not self.mlflow_run_id:
            logger.info(
                "MLflow Run ID not provided. Skipping MLflow artifact fetching."
            )
            return None

        try:
            logger.info(
                f"Attempting to fetch predictions from MLflow Run ID: {self.mlflow_run_id}"
            )
            artifact_uri = f"runs:/{self.mlflow_run_id}/{self.prediction_file_path}"
            local_artifact_path = mlflow.artifacts.download_artifacts(
                artifact_uri=artifact_uri,
            )
            logger.info(
                f"Loading predictions from MLflow artifact path: {local_artifact_path}"
            )
            df = (
                pd.read_csv(local_artifact_path)
                .sort_values(by="prompt_id")
                .set_index("prompt_id")
            )
            return df
        except Exception as e:
            logger.warning(f"Could not fetch predictions from MLflow: {e}")
            return None

    def _get_predictions_from_local(self):
        if not self.prediction_file_path:
            logger.info(
                "Local predictions folder path not provided. Skipping local file loading."
            )
            return None

        if os.path.exists(self.prediction_file_path):
            logger.info(
                f"Loading predictions from local path: {self.prediction_file_path}"
            )
            try:
                df = pd.read_csv(self.prediction_file_path)
                return df
            except Exception as e:
                logger.error(
                    f"Error loading local predictions file '{self.prediction_file_path}': {e}",
                    stack_info=True,
                    exc_info=True,
                )
                return None
        else:
            logger.warning(
                f"Local predictions file/folder not found at: {self.prediction_file_path}"
            )
            return None

    def load_predictions(self):
        if self.mlflow_run_id:
            self.predictions_df = self._get_predictions_from_mlflow()

        if self.predictions_df is None:
            self.predictions_df = self._get_predictions_from_local()

        if self.predictions_df is None:
            logger.error("Failed to load predictions from both MLflow and local paths.")

        required_columns = [
            "context_category",
            "context_identity",
            "answer",
            "prediction",
        ]
        if not all(col in self.predictions_df.columns for col in required_columns):
            raise ValueError(
                f"Missing one or more required columns in predictions file. "
                f"Expected: {required_columns}, Found: {self.predictions_df.columns.tolist()}"
            )

        return self.predictions_df

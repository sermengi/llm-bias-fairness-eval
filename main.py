from pathlib import Path

import mlflow

from src import logger
from src.config import ConfigurationManager
from src.evaluation import ModelEvaluator
from src.inference import ModelInferencePipeline


def main():
    script_dir = Path(__file__).resolve().parent
    configs_dir = script_dir / "configs"
    config_file_path = str(configs_dir / "config.yaml")
    context_config_file_path = str(configs_dir / "context_templates.yaml")

    config_manager = ConfigurationManager(
        config_file_path=config_file_path,
        context_config_file_path=context_config_file_path,
    )

    configs = config_manager.get_all_configurations()
    dataset_config = configs["dataset"]
    model_config = configs["model"]
    artifact_config = configs["artifact"]

    pipeline = ModelInferencePipeline(configs)

    mlflow.set_experiment("LLM Bias and Fairness Evaluation")
    with mlflow.start_run() as run:
        mlflow.log_param("run_id", run.info.run_id)
        logger.info(f"MLflow Run ID: {run.info.run_id}")

        mlflow.log_param("dataset_name", dataset_config.dataset_name)
        mlflow.log_param("dataset_data_files", dataset_config.data_files)
        mlflow.log_param("dataset_split", dataset_config.split)
        mlflow.log_param("dataset_max_samples", dataset_config.max_samples)

        mlflow.log_param("model_name", model_config.model_name)
        mlflow.log_param("model_allowed_choices", model_config.allowed_choices)
        mlflow.log_param("model_batch_size", model_config.batch_size)
        mlflow.log_param(
            "model_tokenizer_padding_side", model_config.tokenizer_padding_side
        )

        pipeline.run_inference()
        mlflow.log_artifact(
            artifact_config.prediction_file_path,
            artifact_path=artifact_config.artifacts_root,
        )
        mlflow.log_artifacts(str(configs_dir), artifact_path="configurations")

        evaluator = ModelEvaluator(artifact_config, mlflow_run_id=run.info.run_id)
        evaluator.load_predictions()


if __name__ == "__main__":
    main()

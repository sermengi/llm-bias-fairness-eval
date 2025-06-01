import mlflow

from src import logger
from src.config import ConfigurationManager
from src.inference import ModelInferencePipeline


def main():
    config = ConfigurationManager(config_file_path="config.yaml")
    dataset_config = config.get_dataset_configuration()
    model_config = config.get_model_configuration()
    artifact_config = config.get_artifact_configuration()

    mlflow.set_experiment("LLM Bias and Fairness Evaluation")
    with mlflow.start_run() as run:
        mlflow.log_param("run_id", run.info.run_id)
        logger.info(f"MLflow Run ID: {run.info.run_id}")

        pipeline = ModelInferencePipeline()
        mlflow.log_param("dataset_name", dataset_config.dataset_name)
        mlflow.log_param("dataset_data_files", dataset_config.data_files)
        mlflow.log_param("dataset_split", dataset_config.split)
        mlflow.log_param("dataset_max_samples", dataset_config.max_samples)

        mlflow.log_text(pipeline.sample_prompt_for_logging, "sample_prompt.txt")

        mlflow.log_param("model_name", model_config.model_name)
        mlflow.log_param("model_allowed_choices", model_config.allowed_choices)

        df = pipeline.run_inference()
        df.to_csv(artifact_config.results_csv_path, index=False)
        mlflow.log_artifact(
            artifact_config.results_csv_path, artifact_path="inference_results"
        )

        mlflow.log_artifact("config.yaml", "configurations")


if __name__ == "__main__":
    main()

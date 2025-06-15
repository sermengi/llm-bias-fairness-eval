import mlflow

from src import logger
from src.config import ConfigurationManager
from src.evaluation import ModelEvaluator
from src.inference import ModelInferencePipeline
from src.paths import CONFIG_FILE_PATH, CONFIGS_DIR, CONTEXT_CONFIG_FILE_PATH


def main():
    config_manager = ConfigurationManager(
        config_file_path=CONFIG_FILE_PATH,
        context_config_file_path=CONTEXT_CONFIG_FILE_PATH,
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

        mlflow.log_params(dict(dataset_config))
        mlflow.log_params(dict(model_config))

        pipeline.run_inference()
        mlflow.log_artifact(
            artifact_config.prediction_file_path,
            artifact_path=artifact_config.artifacts_root,
        )
        mlflow.log_artifacts(str(CONFIGS_DIR), artifact_path="configurations")

        evaluator = ModelEvaluator(artifact_config, mlflow_run_id=run.info.run_id)
        results = evaluator.evaluate()
        flattened_results = {
            f"subgroup_accuracies_{k.replace(' ', '_')}": v
            for k, v in results["subgroup_accuracies"].items()
        }
        mlflow.log_metrics(flattened_results)


if __name__ == "__main__":
    main()

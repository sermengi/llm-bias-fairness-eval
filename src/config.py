from typing import Dict

from pydantic import BaseModel, ValidationError

from src import logger
from src.common import create_directory, read_yaml


class DatasetConfig(BaseModel):
    dataset_name: str
    data_files: str
    split: str
    allowed_choices: list
    max_samples: int


class ModelConfig(BaseModel):
    model_name: str
    batch_size: int
    tokenizer_padding_side: str


class ArtifactConfig(BaseModel):
    artifacts_root: str
    prediction_file_path: str
    mlflow_run_id: str


class ContextConfig(BaseModel):
    base_context: str
    contexts: dict
    identity_formatting: dict


class ConfigurationManager:
    def __init__(self, config_file_path, context_config_file_path):
        self.config = read_yaml(config_file_path)
        self.context_config = read_yaml(context_config_file_path)

    def get_dataset_configuration(self) -> DatasetConfig:
        config = self.config.dataset_configs
        try:
            dataset_config = DatasetConfig(
                dataset_name=config.dataset_name,
                data_files=config.data_files,
                split=config.split,
                allowed_choices=config.allowed_choices,
                max_samples=config.max_samples,
            )
            logger.info("Dataset configurations retrieved successfully.")
        except ValidationError as e:
            logger.error(f"Dataset configuration is not valid: \n{e}")

        return dataset_config

    def get_model_configuration(self) -> ModelConfig:
        config = self.config.model_configs
        try:
            model_config = ModelConfig(
                model_name=config.model_name,
                batch_size=config.batch_size,
                tokenizer_padding_side=config.tokenizer_padding_side,
            )
            logger.info("Model configurations retrieved successfully.")
            return model_config
        except ValidationError as e:
            logger.error(f"Model configuration is not valid: \n{e}")

    def get_artifact_configuration(self) -> ArtifactConfig:
        config = self.config.artifact_configs
        create_directory(config.artifacts_root, verbose=False)
        try:
            artifact_config = ArtifactConfig(
                artifacts_root=config.artifacts_root,
                prediction_file_path=config.prediction_file_path,
                mlflow_run_id=config.mlflow_run_id,
            )
            logger.info("Artifact configurations retrieved successfully.")
            return artifact_config
        except ValidationError as e:
            logger.error(f"Artifact configuration is not valid: \n{e}")

    def get_contexts_configuration(self) -> ContextConfig:
        try:
            context_config = ContextConfig(
                base_context=self.context_config.base_context,
                contexts=self.context_config.contexts,
                identity_formatting=self.context_config.identity_formatting,
            )
            logger.info("Context configurations retrieved successfully.")
            return context_config
        except ValidationError as e:
            logger.error(f"Context configuration is not valid: \n{e}")

    def get_all_configurations(self) -> Dict:
        logger.info("Retrieving all available configurations...")
        dataset_config = self.get_dataset_configuration()
        model_config = self.get_model_configuration()
        artifact_config = self.get_artifact_configuration()
        context_config = self.get_contexts_configuration()

        return {
            "dataset": dataset_config,
            "model": model_config,
            "artifact": artifact_config,
            "context": context_config,
        }

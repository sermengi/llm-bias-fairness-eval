from src.common import read_yaml
from pydantic import BaseModel, ValidationError
from src import logger


class DatasetConfig(BaseModel):
    dataset_name: str
    data_files: str
    split: str
    max_samples: int


class ModelConfig(BaseModel):
    model_name: str
    allowed_choices: list


class ConfigurationManager:
    def __init__(self, config_file_path):
        self.config = read_yaml(config_file_path)

    def get_dataset_configuration(self) -> DatasetConfig:
        config = self.config.dataset_configs
        try:
            dataset_config = DatasetConfig(
                dataset_name=config.dataset_name,
                data_files=config.data_files,
                split=config.split,
                max_samples=config.max_samples,
            )
        except ValidationError as e:
            logger.error(f"Dataset configuration is not valid: \n{e}")

        return dataset_config

    def get_model_configuration(self) -> ModelConfig:
        config = self.config.model_configs
        try:
            model_config = ModelConfig(
                model_name=config.model_name,
                allowed_choices=config.allowed_choices,
            )
            return model_config
        except ValidationError as e:
            logger.error(f"Model configuration is not valid: \n{e}")

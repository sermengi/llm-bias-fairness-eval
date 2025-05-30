import yaml
from box import ConfigBox

from src import logger


def read_yaml(path_to_yaml):
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML FILE: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except Exception as e:
        raise e

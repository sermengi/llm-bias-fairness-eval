from src.config import ConfigurationManager
from src.inference import ModelInferencePipeline


def main():
    pipeline = ModelInferencePipeline()
    df = pipeline.run_inference()

    config = ConfigurationManager(config_file_path="config.yaml")
    artifact_config = config.get_artifact_configuration()
    df.to_csv(artifact_config.results_csv_path, index=False)


if __name__ == "__main__":
    main()

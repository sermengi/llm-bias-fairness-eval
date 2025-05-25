from src.config import ConfigurationManager
from src.data_loader import GSM_MC_PromptBuilder


def main():
    config_file_path = "config.yaml"
    config = ConfigurationManager(config_file_path=config_file_path)
    dataset_config = config.get_dataset_configuration()

    prompt_builder = GSM_MC_PromptBuilder(
        dataset_config.dataset_name,
        data_files=dataset_config.data_files,
        split=dataset_config.split,
        max_samples=dataset_config.max_samples,
    )

    print(prompt_builder.get_sample_prompt(index=0))


if __name__ == "__main__":
    main()

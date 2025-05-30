from src.config import ConfigurationManager
from src.data_loader import GSM_MC_PromptBuilder
from src.models import MultipleChoiceLLM


def main():
    config_file_path = "config.yaml"
    config = ConfigurationManager(config_file_path=config_file_path)
    dataset_config = config.get_dataset_configuration()
    model_config = config.get_model_configuration()

    prompt_builder = GSM_MC_PromptBuilder(
        dataset_config.dataset_name,
        data_files=dataset_config.data_files,
        split=dataset_config.split,
        max_samples=dataset_config.max_samples,
    )

    sample_prompt = prompt_builder.get_sample_prompt(index=0, include_answer=False)
    print(sample_prompt)

    model_name = model_config.model_name
    allowed_choices = model_config.allowed_choices
    model = MultipleChoiceLLM(model_name=model_name, allowed_choices=allowed_choices)
    pred = model.predict(prompt=sample_prompt)
    print("The model prediction: ", pred)


if __name__ == "__main__":
    main()

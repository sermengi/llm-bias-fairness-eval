import pandas as pd
from tqdm import tqdm

from src.config import ConfigurationManager
from src.data_loader import GSM_MC_PromptBuilder
from src.models import MultipleChoiceLLM

CONFIG_FILE_PATH = "config.yaml"


class ModelInferencePipeline:
    def __init__(self):
        self.config = ConfigurationManager(CONFIG_FILE_PATH)
        self.dataset_config = self.config.get_dataset_configuration()
        self.model_config = self.config.get_model_configuration()
        self.initialize_dataset()
        self.initialize_model()

    def initialize_dataset(self):
        self.prompt_builder = GSM_MC_PromptBuilder(
            self.dataset_config.dataset_name,
            data_files=self.dataset_config.data_files,
            split=self.dataset_config.split,
            max_samples=self.dataset_config.max_samples,
        )
        sample_prompt = self.prompt_builder.get_sample_prompt(
            index=0, include_answer=False
        )
        print(sample_prompt)

    def initialize_model(self):
        model_name = self.model_config.model_name
        allowed_choices = self.model_config.allowed_choices
        self.model = MultipleChoiceLLM(
            model_name=model_name, allowed_choices=allowed_choices
        )

    def run_inference(self):
        outputs = self.prompt_builder.generate_prompts_and_metadata()
        results = []
        for sample in tqdm(outputs, desc="Running Inference", total=len(outputs)):
            prompt = sample["prompt"]
            prediction = self.model.predict(prompt)

            results.append(
                {
                    "sample_id": sample["sample_id"],
                    "question": sample["question"],
                    "choice_A": sample["choices"].get("A", ""),
                    "choice_B": sample["choices"].get("B", ""),
                    "choice_C": sample["choices"].get("C", ""),
                    "choice_D": sample["choices"].get("D", ""),
                    "prompt": sample["prompt"],
                    "answer": sample["answer"],
                    "prediction": prediction,
                }
            )

        return pd.DataFrame(results)

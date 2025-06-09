from datasets import load_dataset
from torch.utils.data import Dataset

from src import logger


class GSM_MC_PromptBuilder(Dataset):
    def __init__(
        self, dataset_name, contexts, data_files=None, split="train", max_samples=None
    ):
        self.dataset_name = dataset_name
        self.contexts = contexts
        self.data_files = data_files
        self.split = split
        self.max_samples = max_samples
        self.dataset = None
        self._load_dataset()

        self.processed_data = []
        self._generate_prompts_and_metadata()

    def _load_dataset(self):
        try:
            self.dataset = load_dataset(
                self.dataset_name, data_files=self.data_files, split=self.split
            )
            logger.info(
                f"Dataset {self.dataset_name} (split: {self.split}) is loaded successfully."
            )
        except Exception as e:
            logger.error(f"Failed to load dataset {self.dataset_name}. Error: {e}")
            raise ValueError(
                "Please check the dataset configurations: dataset name, split, or file path."
            )

        if self.max_samples is not None:
            if self.max_samples > len(self.dataset):
                logger.warning(
                    f"{self.dataset_name} doesn't have {self.max_samples} samples in {self.split} split. Collecting all the available samples: {len(self.dataset)}"
                    f"Using all available samples: {len(self.dataset)}"
                )
            else:
                logger.info(f"Successfully retrieved {self.max_samples} samples")
                self.dataset = self.dataset.select(range(self.max_samples))

    def format_sample(self, sample, context=None, answer=None):
        context = context or sample.get("Context", "").strip()
        question = sample["Question"]
        choices = {k: str(v) for k, v in sample.items() if k in ["A", "B", "C", "D"]}
        choice_list = "\n".join(
            [f"{option}. {choice}" for option, choice in choices.items()]
        )

        prompt = f"{context}\n\nQuestion: {question}\n\nChoices:\n{choice_list}"

        if answer is not None:
            prompt += f"\n\nAnswer: {answer}"

        return prompt

    def get_sample_prompt(self, index, context=None, include_answer=True):
        try:
            sample = self.dataset[index]
        except IndexError as e:
            logger.error(
                f"Index {index} is out of bounds for dataset of size {len(self.dataset)}."
            )
            raise e

        answer = sample["Answer"] if include_answer else None
        prompt = self.format_sample(sample=sample, context=context, answer=answer)
        return prompt

    def _generate_prompts_and_metadata(self):
        logger.info("Generating prompts and metadata for all samples...")
        outputs = []
        prompt_id = 0

        for idx, sample in enumerate(self.dataset):
            answer = sample["Answer"]
            question = sample["Question"]
            choices = {k: sample.get(k, "") for k in ["A", "B", "C", "D"]}

            # generate a baseline prompt
            if prompt_id % len(self.dataset) == 0:
                prompt = self.format_sample(sample, context="", answer=None)

                item = {
                    "prompt_id": prompt_id,
                    "sample_id": idx,
                    "question": question,
                    "choices": choices,
                    "prompt": prompt,
                    "answer": answer,
                    "context_info": {"category": "baseline", "identity": "none"},
                }
                outputs.append(item)
                prompt_id += 1

            for category, contexts_in_category in self.contexts.items():
                for identity, context_prompt in contexts_in_category.items():
                    prompt = self.format_sample(
                        sample, context=context_prompt, answer=None
                    )

                    item = {
                        "prompt_id": prompt_id,
                        "sample_id": idx,
                        "question": question,
                        "choices": choices,
                        "prompt": prompt,
                        "answer": answer,
                        "context_info": {"category": category, "identity": identity},
                    }
                    outputs.append(item)
                    prompt_id += 1

        self.processed_data = outputs
        logger.info(f"Generated {len(self.processed_data)} prompts.")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.processed_data):
            raise IndexError(
                f"Index {idx} is out of bounds for dataset of size {len(self.processed_data)}."
            )
        return self.processed_data[idx]

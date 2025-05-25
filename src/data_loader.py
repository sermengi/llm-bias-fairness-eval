from datasets import load_dataset


class GSM_MC_PromptBuilder:
    def __init__(self, dataset_name, data_files=None, split="train", max_samples=None):
        self.dataset_name = dataset_name
        self.data_files = data_files
        self.split = split
        self.max_samples = max_samples
        self.dataset = None
        self._load_dataset()

    def _load_dataset(self):
        try:
            self.dataset = load_dataset(
                self.dataset_name, data_files=self.data_files, split=self.split
            )
        except Exception as e:
            raise e

        if self.max_samples is not None:
            if self.max_samples > len(self.dataset):
                print("Not enough sample in dataset")
            else:
                print("Successfully retrieved specified number of samples")
                self.dataset = self.dataset.select(range(self.max_samples))

    def format_sample(self, sample, answer=None):
        context = sample.get("context", "").strip()
        question = sample["Question"]
        choices = {k: str(v) for k, v in sample.items() if k in ["A", "B", "C", "D"]}
        choice_list = "\n".join(
            [f"{option}. {choice}" for option, choice in choices.items()]
        )

        prompt = f"{context}\n\nQuestion: {question}\n\nChoices:\n{choice_list}"

        if answer is not None:
            prompt += f"\n\nAnswer: {answer}"

        return prompt

    def get_sample_prompt(self, index):
        try:
            sample = self.dataset[index]
        except IndexError:
            raise IndexError(
                f"Index {index} is out of bounds for dataset of size {len(self.dataset)}."
            )

        prompt = self.format_sample(sample=sample, answer=sample["Answer"])
        return prompt

    def generate_prompts(self):
        return [self.format_sample(sample) for sample in self.dataset]

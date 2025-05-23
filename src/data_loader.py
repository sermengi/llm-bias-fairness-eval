from datasets import load_dataset


class PromptBuilder:
    def __init__(self, dataset_name, data_files=None, split="train", max_samples=None):
        self.dataset = load_dataset(dataset_name, data_files=data_files, split=split)
        if max_samples:
            self.dataset = self.dataset.select(range(max_samples))

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
        sample = self.dataset[index]
        prompt = self.format_sample(sample=sample, answer=sample["Answer"])
        return prompt

    def get_prompts(self):
        return [self.format_sample(sample) for sample in self.dataset]

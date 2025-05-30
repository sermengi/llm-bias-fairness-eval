import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import logger


class MultipleChoiceLLM:
    def __init__(self, model_name, allowed_choices=None):
        try:
            self.device = xm.xla_device()
            logger.info(f"TPU device acquired: {self.device}")
        except RuntimeError as e:
            logger.error(f"Error acquiring TPU: {e}")
            logger.error("Falling back to CPU. Inference will be much slower.")
            self.device = torch.device("cpu")

        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=None, torch_dtype=torch.bfloat16
        )

        if self.device.type == "xla":
            self.model = self.model.to(self.device)
            logger.info("Model successfully moved to TPU.")
        else:
            logger.info("Model remains on CPU.")

        self.allowed_choices = allowed_choices or ["A", "B", "C", "D"]
        self.allowed_token_ids = self._get_allowed_token_ids()

    def _get_allowed_token_ids(self):
        return [
            self.tokenizer.encode(choice, add_special_tokens=False)[0]
            for choice in self.allowed_choices
        ]

    def predict(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits  # shape: [1, seq_len, vocab_size]

        last_token_logits = logits[:, -1, :]  # shape: [1, vocab_size]

        restricted_logits = torch.full_like(last_token_logits, float("-inf"))
        restricted_logits[:, self.allowed_token_ids] = last_token_logits[
            :, self.allowed_token_ids
        ]

        predicted_token_id = torch.argmax(restricted_logits, dim=-1).unsqueeze(-1)
        predicted_choice = self.tokenizer.decode(
            predicted_token_id.squeeze(), skip_special_tokens=True
        )

        return predicted_choice

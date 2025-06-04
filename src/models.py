from typing import List, Union

import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import logger


class MultipleChoiceLLM:
    def __init__(
        self,
        model_name: str,
        allowed_choices: List[str] = None,
        tokenizer_padding_side: str = "left",
    ):
        self.process_ordinal_str = "N/A (CPU or XLA not fully initialized)"
        try:
            self.device = xm.xla_device()
            if xm.xla_device_hw(str(self.device)):
                self.process_ordinal_str = f"Ordinal {xm.get_local_ordinal()}"
            logger.info(
                f"[{self.process_ordinal_str}] TPU device acquired: {self.device}"
            )
        except RuntimeError as e:
            logger.error(f"Error acquiring TPU: {e}")
            logger.error("Falling back to CPU. Inference will be much slower.")
            self.device = torch.device("cpu")
            logger.info(f"[{self.process_ordinal_str}] Using CPU device: {self.device}")

        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(
                f"[{self.process_ordinal_str}] Tokenizer pad_token set to eos_token: '{self.tokenizer.eos_token}' (ID: {self.tokenizer.eos_token_id})"
            )
        self.tokenizer.padding_side = tokenizer_padding_side

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=None, torch_dtype=torch.bfloat16
        )
        self.model.eval()

        if self.device.type == "xla":
            self.model = self.model.to(self.device)
            logger.info(
                f"[{self.process_ordinal_str}] Model successfully moved to TPU ({self.device})."
            )
        else:
            logger.info(f"[{self.process_ordinal_str}] Model remains on CPU.")

        self.allowed_choices = allowed_choices or ["A", "B", "C", "D"]
        self.allowed_token_ids = self._get_allowed_token_ids()
        self.allowed_token_id_to_choice_map = {
            token_id: choice
            for token_id, choice in zip(self.allowed_token_ids, self.allowed_choices)
        }

    def _get_allowed_token_ids(self):
        token_ids = []
        for choice in self.allowed_choices:
            encoded = self.tokenizer.encode(choice, add_special_tokens=False)
            if len(encoded) == 1:
                token_ids.append(encoded[0])
            else:
                logger.error(
                    f"[{self.process_ordinal_str}] Choice '{choice}' tokenized into multiple IDs: {encoded}. "
                    f"The current prediction logic expects choices to be single tokens. "
                    f"Using the first token ID: {encoded[0]}"
                )
                token_ids.append(encoded[0])
        return token_ids

    def predict(self, prompts: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(prompts, str):
            is_single_prompt = True
            prompts_batch = [prompts]
        elif isinstance(prompts, list):
            is_single_prompt = False
            prompts_batch = prompts
        else:
            logger.error("Input 'prompts' must be a string or a list of strings.")
            raise TypeError

        inputs = self.tokenizer(
            prompts_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).to(self.device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]

        if self.tokenizer.padding_side == "right":
            sequence_lengths = attention_mask.sum(dim=1) - 1
            last_token_logits = logits[
                torch.arange(logits.shape[0], device=self.device), sequence_lengths, :
            ]
        else:
            last_token_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]

        restricted_logits = torch.full_like(last_token_logits, float("-inf"))
        allowed_token_ids_tensor = torch.tensor(
            self.allowed_token_ids, device=self.device
        )
        restricted_logits[:, allowed_token_ids_tensor] = last_token_logits[
            :, allowed_token_ids_tensor
        ]

        predicted_token_ids_batch = torch.argmax(
            restricted_logits, dim=-1
        )  # Shape: [batch_size]

        predicted_choices = []
        for token_id in predicted_token_ids_batch:
            choice = self.allowed_token_id_to_choice_map.get(token_id.item())
            if choice is not None:
                predicted_choices.append(choice)
            else:
                fallback_decode = self.tokenizer.decode(
                    token_id.item(), skip_special_tokens=True
                )
                logger.warning(
                    f"[{self.process_ordinal_str}] Predicted token ID {token_id.item()} not found in allowed_token_id_to_choice_map. "
                    f"Fallback decode: '{fallback_decode}'"
                )
                predicted_choices.append(fallback_decode)

        if is_single_prompt:
            return predicted_choices[0]
        else:
            return predicted_choices

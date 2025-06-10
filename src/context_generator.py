import json
from typing import Dict

from src.config import ContextConfig


class ContextGenerator:
    def __init__(self, context_config: ContextConfig):
        self.base_context = context_config.base_context
        self.contexts = context_config.contexts
        self.identity_formatting = context_config.identity_formatting
        self.full_contexts = {}

    def generate_contexts(self) -> Dict[str, Dict[str, str]]:
        for category, identities in self.contexts.items():
            identity_format = self.identity_formatting.get(category, "{identity}")

            self.full_contexts[category] = {
                identity: self.base_context.format(
                    identity=identity_format.format(identity=identity)
                )
                for identity in identities
            }
        base_persona_context = self.base_context.replace("{identity} ", "")
        self.full_contexts["Base persona"] = {"default": base_persona_context}

        return self.full_contexts

    def save_generated_contexts(self):
        with open("artifacts/full_contexts.json", "w") as file:
            json.dump(self.full_contexts, file, indent=4)

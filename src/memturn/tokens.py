"""Budget counters: the offline counter is deliberately not called an LLM tokenizer."""
import re
from typing import Protocol


class TokenCounter(Protocol):
    name: str
    def count(self, text: str) -> int: ...


class OfflineCounter:
    name = "offline_lexical_not_model_tokens"

    def count(self, text):
        return len(re.findall(r"[A-Za-z0-9_]+|[^\s]", text))


class HuggingFaceCounter:
    def __init__(self, model_id, revision=None, tokenizer=None):
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.tokenizer = tokenizer
        self.name = f"huggingface:{model_id}@{revision or 'unfrozen'}"

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

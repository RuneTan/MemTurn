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


class TokenizerCounter:
    """Count with an explicitly supplied tokenizer; never load model assets."""
    def __init__(self, tokenizer, name):
        self.tokenizer, self.name = tokenizer, name
    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

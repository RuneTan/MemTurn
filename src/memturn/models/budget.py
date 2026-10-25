"""Exact local chat-template accounting and crash-conservative run quotas."""
class TokenBudgetClient:
    def __init__(self, client, tokenizer, store, role, total_tokens, context_window):
        self.client, self.tokenizer, self.store = client, tokenizer, store
        self.role, self.total_tokens, self.context_window = role, total_tokens, context_window
        self.config = client.config

    @property
    def usage(self):
        return self.client.usage

    def complete(self, messages, model_override=None):
        options = self.config.chat_template_kwargs or {}
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **options)
        prompt_tokens = len(ids)
        reservation = prompt_tokens + self.config.max_tokens
        if reservation > self.context_window:
            raise ValueError('chat prompt plus output reservation exceeds model context')
        self.store.reserve_compute(self.role, reservation, self.total_tokens)
        result = self.client.complete(messages, model_override)
        usage = self.client.response_usage.value
        if usage.get('prompt_tokens') != prompt_tokens or 'completion_tokens' not in usage:
            raise ValueError('server token accounting differs from the frozen local chat template')
        self.store.settle_compute(self.role, reservation, prompt_tokens + usage['completion_tokens'])
        return result

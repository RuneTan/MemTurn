"""Small exploratory-run helpers; metrics and audits are never agent memory."""
from collections import deque
import time

from .models.http import ModelHTTPError


class ExperimentPaused(RuntimeError):
    pass


class BudgetedChat:
    def __init__(self, client, audit, token_budget=180000, prior_usage=None):
        self.client, self.audit, self.token_budget = client, audit, token_budget
        if prior_usage:
            self.client.usage.update(prior_usage)
        self.window = deque()
        self.paused = None
        self.starting_tokens = self.charged_tokens

    @property
    def usage(self):
        return self.client.usage

    @property
    def charged_tokens(self):
        u = self.usage
        return u["prompt_tokens"] + u["completion_tokens"] - u.get("cached_prompt_tokens", 0)

    def can_start(self):
        return self.budget_used + 4096 <= self.token_budget

    @property
    def budget_used(self):
        return self.charged_tokens - self.starting_tokens

    def complete(self, messages, model_override=None):
        if not self.can_start():
            self.paused = "configured_writer_token_budget"
            raise ExperimentPaused(self.paused)
        # Leave headroom below the published 8k TPM; actual server limits still win.
        for attempt in range(4):
            now = time.monotonic()
            while self.window and now - self.window[0][0] >= 60:
                self.window.popleft()
            if self.window and sum(n for _, n in self.window) + 2000 > 6500:
                wait = min(30.0, max(0.1, 60 - (now - self.window[0][0])))
                self.audit({"event": "writer_throttle", "seconds": wait})
                time.sleep(wait)
                continue
            before = self.charged_tokens
            try:
                result = self.client.complete(messages, model_override)
                self.window.append((time.monotonic(), self.charged_tokens - before))
                return result
            except ModelHTTPError as error:
                if error.daily or error.status in (401, 403):
                    self.paused = "provider_daily_quota" if error.daily else "provider_authentication"
                    raise ExperimentPaused(self.paused) from None
                if error.status == 429 or error.status >= 500:
                    if attempt < 3:
                        wait = min(30.0, max(10.0, error.retry_after))
                        self.audit({"event": "writer_retry", "status": error.status, "seconds": wait})
                        time.sleep(wait)
                        continue
                    self.paused = "provider_rate_or_service_limit"
                    raise ExperimentPaused(self.paused) from None
                raise
        self.paused = "writer_throttle_checkpoint"
        raise ExperimentPaused(self.paused)


class ResumableWriter:
    def __init__(self, writer):
        self.writer, self.deferred = writer, None

    def write(self, episode, batch):
        try:
            return self.writer.write(episode, batch)
        except ExperimentPaused:
            self.deferred = episode
            raise


def grouped_tasks(count, group_size=20, seed=42):
    from .contracts import Task
    conditions = ("vault:v1", "api:v1", "vault:v2", "api:v2")
    for index in range(count):
        family, version = conditions[(index // group_size) % len(conditions)].split(":")
        yield Task(f"chain_{seed}_{index:04}", family, version, seed + index)


def probe_actions(backend, store, policy, counter, config, checkpoint):
    """Paired first actions on fixed held-out objects; no feedback or state writes."""
    from .contracts import RetrievalResult, Task
    from .memory.retrieval import Retriever
    from .tasks.toy import ToyEnvironment, PREPARE
    rows = []
    retriever = Retriever(counter, config)
    with store.session() as view:
        for condition_index, condition in enumerate(PREPARE):
            family, version = condition.split(":")
            for sample in range(2):
                env = ToyEnvironment()
                task = Task(f"heldout_{condition_index}_{sample}", family, version,
                            1000000 + condition_index * 10 + sample)
                observation = env.reset(task)
                retrieved = retriever.search(view.memories, observation, applicability=env.retrieval_scope(observation))
                for label, model_version, memories in (
                    ("base_no_memory", "base", ()),
                    ("adapter_no_memory", view.model_version, ()),
                    ("base_current_memory", "base", retrieved),
                    ("adapter_current_memory", view.model_version, retrieved)):
                    error = None
                    try:
                        remaining = config.max_extra_searches if label.endswith("current_memory") else 0
                        action = policy.act(observation, memories, model_version,
                                            (RetrievalResult(None, memories, remaining),))
                    except Exception as exception:
                        action, error = None, type(exception).__name__
                    rows.append({"checkpoint": checkpoint, "condition": condition, "sample": sample,
                                 "regime": label, "action": action, "expected": PREPARE[condition],
                                 "correct": action == PREPARE[condition], "error": error,
                                 "memory_refs": [m.ref for m in memories], "model_version": model_version})
                env.close()
    return rows

"""Immutable, JSON-serializable contracts. No model or environment imports."""
from dataclasses import asdict, dataclass, field
import json
from typing import Any


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Support:
    task_id: str
    step: int


@dataclass(frozen=True)
class Example:
    task_id: str
    step: int
    prompt: str
    target: str
    input_step: int | None = None


@dataclass(frozen=True)
class Memory:
    id: str
    content: str
    applicability: str
    supports: tuple[Support, ...]
    examples: tuple[Example, ...] = ()
    parents: tuple[str, ...] = ()
    revision: int = 1
    created_batch: int = 0
    recency_anchor: int = 0
    consolidated: bool = False

    @property
    def ref(self):
        return f"{self.id}@{self.revision}"

    @property
    def payload(self):
        return f"{self.applicability}\n{self.content}"

    @property
    def task_ids(self):
        return frozenset(s.task_id for s in self.supports)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        data["supports"] = tuple(Support(**s) for s in data.get("supports", ()))
        data["examples"] = tuple(Example(**e) for e in data.get("examples", ()))
        data["parents"] = tuple(data.get("parents", ()))
        return cls(**data)


@dataclass(frozen=True)
class Step:
    observation: str
    action: str
    feedback: str
    accepted: bool


@dataclass(frozen=True)
class RetrievalResult:
    """Transient policy transcript, separate from environment evidence indices."""
    query: str | None
    memories: tuple[Memory, ...]
    remaining: int
    error: str | None = None


@dataclass(frozen=True)
class Episode:
    task_id: str
    family: str
    condition: str
    steps: tuple[Step, ...]
    reward: float
    success: bool
    model_version: str = "base"
    elapsed_seconds: float = 0.0
    status: str = "completed"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        data["steps"] = tuple(Step(**s) for s in data["steps"])
        return cls(**data)


@dataclass(frozen=True)
class Task:
    id: str
    family: str
    version: str
    seed: int = 0
    max_steps: int = 8
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Transition:
    observation: str
    reward: float
    done: bool
    accepted: bool = False
    success: bool = False


@dataclass(frozen=True)
class Snapshot:
    lease_id: str
    version: int
    batch: int
    model_version: str
    memories: tuple[Memory, ...]


@dataclass(frozen=True)
class Artifact:
    version: str
    path: str
    backend: str
    parent_version: str
    metrics: dict

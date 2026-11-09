from typing import Protocol
from ..contracts import Artifact, Example, Memory, Step


class Learner(Protocol):
    def train(self, examples: tuple[Example, ...], parent_version: str, *,
              histories: dict[tuple[str, int], tuple[Step, ...]],
              memory_context: tuple[Memory, ...] = ()) -> Artifact: ...
    def prepare(self, artifact: Artifact) -> None: ...
    def retain(self, versions: set[str]) -> None: ...

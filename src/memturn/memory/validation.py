"""Lightweight provenance, material binding and source fidelity; no semantic judge."""
from ..contracts import Support
from ..learning.materials import covered_text, unique_examples


class Validator:
    def __init__(self, store):
        self.store = store

    def _source_exists(self, episode, ref):
        return bool(episode and isinstance(ref.task_id, str) and ref.task_id
                    and ref.task_id == episode.task_id
                    and isinstance(ref.step, int) and not isinstance(ref.step, bool)
                    and 0 <= ref.step < len(episode.steps))

    def _episode(self, ref):
        if not isinstance(ref.task_id, str) or not ref.task_id:
            return None
        return self.store.episode(ref.task_id)

    def source_coverage_count(self, memory):
        return len({ref.task_id for ref in memory.supports
                    if self._source_exists(self._episode(ref), ref)})

    def valid_memory(self, memory):
        try:
            return bool(isinstance(memory.content, str) and memory.content.strip()
                        and isinstance(memory.applicability, str) and memory.applicability.strip()
                        and memory.supports and all(self._source_exists(self._episode(s), s)
                                                   for s in memory.supports))
        except (TypeError, AttributeError):
            return False

    def training_examples(self, memory):
        examples = unique_examples(memory.examples)
        if self.store.config.training_material == 'decision_only':
            return tuple(e for e in examples if e.example_type == 'decision')
        if self.store.config.training_material == 'qa_only':
            return tuple(e for e in examples if e.example_type == 'rule_qa')
        return examples

    def valid_example(self, example, memory):
        try:
            return self._valid_example(example, memory)
        except (AttributeError, KeyError, TypeError, ValueError):
            return False

    def _valid_example(self, example, memory):
        if example.memory_ref != memory.ref or not example.prompt.strip() or not example.target.strip():
            return False
        try:
            taught = covered_text(memory, example.covered_content_refs)
        except (TypeError, ValueError):
            return False
        if example.example_type == 'rule_qa':
            if example.output_kind != 'text' or example.target != taught:
                return False
            if example.task_id or example.step != -1 or example.input_step is not None:
                return False
            return bool(example.support_refs) and set(example.support_refs) <= set(memory.supports) and all(
                self._source_exists(self._episode(s), s) for s in example.support_refs)
        if example.example_type != 'decision' or example.output_kind != 'environment_action':
            return False
        if (example.input_step is not None and (isinstance(example.input_step, bool)
                or not isinstance(example.input_step, int) or example.input_step != example.step)):
            return False  # No source-state projection in the current protocol.
        ref = Support(example.task_id, example.step)
        if (ref not in memory.supports or not example.support_refs
                or any(s != ref for s in example.support_refs)):
            return False
        episode = self.store.episode(example.task_id)
        try:
            return bool(self._source_exists(episode, ref)
                        and (episode.steps[example.step].accepted
                             or episode.steps[example.step].evidence.get("action_admissible") is True)
                        and example.prompt == episode.steps[example.step].observation
                        and example.target == episode.steps[example.step].action)
        except Exception:
            return False

    def trainable(self, memory):
        examples = self.training_examples(memory)
        if not self.valid_memory(memory) or not examples or len(examples) > self.store.config.max_examples_per_job:
            return False
        if self.store.config.training_material == 'mixed' and {e.example_type for e in examples} != {'rule_qa','decision'}:
            return False
        return all(self.valid_example(e, memory) for e in examples)

    def validate_replacement(self, region, outputs):
        parents = {m.ref: m for m in region}
        return bool(outputs) and len({m.id for m in outputs}) == len(outputs) and all(
            m.parents and set(m.parents) <= set(parents)
            and m.task_ids <= set().union(*(parents[ref].task_ids for ref in m.parents))
            and self.valid_memory(m) for m in outputs)

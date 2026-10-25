"""Source truth is delegated; mixed-material binding and fidelity are checked here."""
from ..contracts import Support
from ..learning.materials import covered_text


class Validator:
    def __init__(self, store, verifier):
        self.store, self.verifier = store, verifier

    def _supports(self, memory, episode, ref):
        try:
            return bool(episode and ref.task_id == episode.task_id
                        and isinstance(ref.step, int) and not isinstance(ref.step, bool)
                        and 0 <= ref.step < len(episode.steps)
                        and self.verifier.verify_support(memory, episode, ref))
        except Exception:
            return False

    def support_count(self, memory):
        supported = set()
        for ref in memory.supports:
            episode = self.store.episode(ref.task_id)
            if self._supports(memory, episode, ref):
                supported.add(ref.task_id)
        return len(supported)

    def valid_memory(self, memory):
        return bool(memory.content.strip() and memory.applicability.strip() and memory.supports
                    and all(self._supports(memory, self.store.episode(s.task_id), s) for s in memory.supports))

    def training_examples(self, memory):
        if self.store.config.training_material == 'decision_only':
            return tuple(e for e in memory.examples if e.example_type == 'decision')
        if self.store.config.training_material == 'qa_only':
            return tuple(e for e in memory.examples if e.example_type == 'rule_qa')
        return memory.examples

    def valid_example(self, example, memory):
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
            # Structural leakage check; semantic paraphrase leakage still needs domain review.
            compact = lambda s: ' '.join(s.casefold().split())
            if compact(taught) in compact(example.prompt):
                return False
            return bool(example.support_refs) and set(example.support_refs) <= set(memory.supports) and all(
                self._supports(memory, self.store.episode(s.task_id), s) for s in example.support_refs)
        if example.example_type != 'decision' or example.output_kind != 'environment_action':
            return False
        if example.input_step not in (None, example.step):
            return False  # No source-state projection in the current protocol.
        ref = Support(example.task_id, example.step)
        if ref not in memory.supports or example.support_refs != (ref,):
            return False
        episode = self.store.episode(example.task_id)
        try:
            return bool(self._supports(memory, episode, ref)
                        and self.verifier.verify_example(example, episode))
        except Exception:
            return False

    def trainable(self, memory):
        examples = self.training_examples(memory)
        if not self.valid_memory(memory) or not examples or len(examples) > self.store.config.max_examples_per_job:
            return False
        if self.store.config.training_material == 'mixed' and {e.example_type for e in examples} != {'rule_qa','decision'}:
            return False
        if self.store.config.training_material == 'mixed' and sum(e.example_type == 'rule_qa' for e in examples) != 1:
            return False
        if self.store.config.training_material == 'qa_only' and len(examples) != 1:
            return False
        return all(self.valid_example(e, memory) for e in examples)

    def validate_replacement(self, region, outputs):
        parents = {m.ref: m for m in region}
        return bool(outputs) and len({m.id for m in outputs}) == len(outputs) and all(
            m.parents and set(m.parents) <= set(parents)
            and m.task_ids <= set().union(*(parents[ref].task_ids for ref in m.parents))
            and self.valid_memory(m) for m in outputs)

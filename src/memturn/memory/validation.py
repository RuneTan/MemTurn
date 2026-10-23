"""Evidence references are checked by a task-specific verifier, not self-certified."""
class Validator:
    def __init__(self, store, verifier):
        self.store, self.verifier = store, verifier

    def support_count(self, memory):
        supported = set()
        for ref in memory.supports:
            episode = self.store.episode(ref.task_id)
            if episode and self.verifier.verify_support(memory, episode, ref):
                supported.add(ref.task_id)
        return len(supported)

    def valid_memory(self, memory):
        return bool(memory.content.strip() and memory.applicability.strip() and memory.supports
                    and all((ep := self.store.episode(s.task_id)) is not None
                            and self.verifier.verify_support(memory, ep, s) for s in memory.supports))

    def trainable(self, memory):
        return self.valid_memory(memory) and bool(memory.examples) and all(
            e.task_id in memory.task_ids and (ep := self.store.episode(e.task_id)) is not None
            and self.verifier.verify_example(e, ep) for e in memory.examples)

    def validate_replacement(self, region, outputs):
        parents = {m.ref: m for m in region}
        return bool(outputs) and len({m.id for m in outputs}) == len(outputs) and all(
            m.parents and set(m.parents) <= set(parents)
            and m.task_ids <= set().union(*(parents[ref].task_ids for ref in m.parents))
            and self.valid_memory(m) for m in outputs)

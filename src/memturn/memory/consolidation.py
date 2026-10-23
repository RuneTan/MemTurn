"""Offline fixture implementations. Model-backed implementations use the same contracts."""
from collections import defaultdict
import hashlib
from ..contracts import Example, Memory, Support


class ToyWriter:
    def write(self, episode, batch):
        from ..tasks.toy import PREPARE
        for index, step in enumerate(episode.steps):
            if step.accepted and step.action == PREPARE.get(episode.condition):
                text = (f"For {episode.condition}, first call {step.action}. "
                        f"Observed in {episode.task_id}; actual feedback was {step.feedback}. "
                        + "Preserve the environment version and the required preparation before completion. " * 6)
                return Memory(f"raw_{episode.task_id}", text, episode.condition,
                              (Support(episode.task_id, index),),
                              (Example(episode.task_id, index, step.observation, step.action),),
                              created_batch=batch, recency_anchor=batch)
        return None


class ToyConsolidator:
    def __init__(self, store):
        self.store = store

    def consolidate(self, region, mode, batch, output_budget):
        from ..tasks.toy import ToyVerifier
        grouped = defaultdict(list)
        for memory in region:
            grouped[memory.applicability].append(memory)
        outputs = []
        for condition, parents in sorted(grouped.items()):
            refs = tuple(sorted({s for m in parents for s in m.supports}, key=lambda s: (s.task_id, s.step)))
            examples = []
            for ref in refs:
                ep = self.store.episode(ref.task_id)
                if ep and 0 <= ref.step < len(ep.steps):
                    step = ep.steps[ref.step]
                    clean = Example(ref.task_id, ref.step, ep.steps[0].observation, step.action, input_step=0)
                    if ToyVerifier().verify_example(clean, ep):
                        examples.append(clean)
                    else:
                        examples.append(Example(ref.task_id, ref.step, step.observation, step.action))
            if not examples:
                continue
            actions = {e.target for e in examples}
            if len(actions) != 1:
                continue
            action = examples[0].target
            digest = hashlib.sha256("|".join(m.ref for m in parents).encode()).hexdigest()[:16]
            outputs.append(Memory(f"summary_{digest}", f"For {condition}, first call {action}.",
                                  condition, refs, tuple(examples[:32]) if mode == "regular" else (),
                                  tuple(m.ref for m in parents), created_batch=batch,
                                  recency_anchor=max(m.recency_anchor for m in parents), consolidated=True))
        return tuple(outputs)

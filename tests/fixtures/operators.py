"""Offline fixture implementations. Model-backed implementations use the same contracts."""
from collections import defaultdict
from dataclasses import replace
import hashlib
from memturn.contracts import Example, Memory, Support
from memturn.learning.materials import bind_example, rule_question


class ToyWriter:
    def write(self, episode, batch):
        from tests.fixtures.environment import PREPARE
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
        grouped = defaultdict(list)
        for memory in region:
            grouped[memory.applicability].append(memory)
        outputs, remaining = [], self.store.config.max_examples_per_job
        for condition, parents in sorted(grouped.items()):
            refs = tuple(sorted({s for m in parents for s in m.supports}, key=lambda s: (s.task_id, s.step)))
            examples = []
            for ref in refs:
                ep = self.store.episode(ref.task_id)
                if ep and 0 <= ref.step < len(ep.steps):
                    step = ep.steps[ref.step]
                    examples.append(Example(ref.task_id, ref.step, step.observation, step.action))
            if not examples:
                continue
            actions = {e.target for e in examples}
            if len(actions) != 1:
                continue
            action = examples[0].target
            digest = hashlib.sha256("|".join(m.ref for m in parents).encode()).hexdigest()[:16]
            memory = Memory(f"summary_{digest}", f"For {condition}, first call {action}.",
                                  condition, refs, (),
                                  tuple(m.ref for m in parents), created_batch=batch,
                                  recency_anchor=max(m.recency_anchor for m in parents), consolidated=True)
            if mode == 'regular' and remaining >= 2:
                decisions = tuple(bind_example(e, memory) for e in examples[:remaining-1])
                material = (rule_question(memory), *decisions)
                memory = replace(memory, examples=material)
                remaining -= len(material)
            outputs.append(memory)
        return tuple(outputs)

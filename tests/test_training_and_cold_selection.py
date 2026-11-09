"""Full learner inputs and cold selection bounded by tokens rather than item count."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from memturn.config import Config
from memturn.contracts import Artifact, Example, Memory, Support
from memturn.learning.materials import rule_question
from memturn.memory.policies import cold_region
from memturn.memory.store import Store
from memturn.memory.validation import Validator
from memturn.runtime.service import Runtime
from memturn.tokens import OfflineCounter
from tests.fixtures.operators import ToyConsolidator, ToyWriter
from tests.test_core import episode


class RecordingLearner:
    def __init__(self):
        self.calls = []

    def train(self, examples, parent_version, *, histories, memory_context=()):
        self.calls.append((examples, parent_version, histories, memory_context))
        return Artifact('v1', 'fixture', 'test-only', parent_version, {})

    def prepare(self, artifact): pass
    def retain(self, versions): pass


class TrainingAndColdSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name)/'state.sqlite', Config(), OfflineCounter())
        self.validator = Validator(self.store)
        self.runtime = None

    def tearDown(self):
        if self.runtime:
            self.runtime.close()
        self.store.close()
        self.temp.cleanup()

    def add(self, name, **changes):
        ep = episode(name)
        self.store.record_result(ep)
        memory = replace(ToyWriter().write(ep, self.store.batch), **changes)
        self.store.append(memory)
        self.store.finish_source(name)
        return memory

    def candidate(self, step=1):
        a, b = self.add('a'), self.add('b')
        ep = self.store.episode('a')
        memory = replace(a, id='selected', content='Reusable general procedure',
                         applicability='Task conditions', parents=(a.ref, b.ref),
                         supports=(Support('a',step), Support('b',0)), consolidated=True, examples=())
        decision = Example('a', step, ep.steps[step].observation, ep.steps[step].action,
                           memory_ref=memory.ref, covered_content_refs=(memory.ref+'#content',),
                           support_refs=(Support('a',step),))
        memory = replace(memory, examples=(rule_question(memory), decision))
        self.store.append(memory)
        self.assertTrue(self.validator.trainable(memory))
        return memory

    def train(self, selected, learner=None):
        learner = learner or RecordingLearner()
        self.runtime = Runtime(self.store, ToyWriter(), ToyConsolidator(self.store),
                               self.validator, learner)
        self.runtime._train('context_probe', selected)
        return learner

    def test_original_probe_f_delivers_history_without_capability_flag(self):
        selected = self.candidate()
        expected = self.store.episode('a').steps[:1]
        learner = self.train(selected)
        self.assertEqual(len(learner.calls), 1)
        examples, parent, histories, memories = learner.calls[0]
        self.assertEqual(examples, selected.examples)
        self.assertEqual(parent, 'base')
        self.assertEqual(histories, {('a',1):expected})
        self.assertEqual(memories, ())
        self.assertEqual(self.store.model_version, 'v1')

    def test_first_step_decision_has_explicit_empty_history(self):
        learner = self.train(self.candidate(step=0))
        self.assertEqual(learner.calls[0][2], {('a',0):()})
        self.assertEqual(self.store.model_version, 'v1')

    def test_qa_only_uses_same_interface_with_empty_history_map(self):
        selected = self.candidate()
        self.store.config = replace(self.store.config, training_material='qa_only')
        learner = self.train(selected)
        self.assertEqual(learner.calls[0][0], selected.examples[:1])
        self.assertEqual(learner.calls[0][2:], ({}, ()))
        self.assertEqual(self.store.model_version, 'v1')

    def test_unrelated_memory_is_delivered_alongside_native_history(self):
        selected = self.candidate()
        allowed = self.add('c', content='Additional context', applicability='Task conditions')
        learner = self.train(selected)
        self.assertEqual(learner.calls[0][3], (allowed,))
        self.assertEqual(learner.calls[0][2], {('a',1):self.store.episode('a').steps[:1]})

    def test_outdated_learner_fails_without_fallback_or_publication(self):
        selected = self.candidate()
        class OutdatedLearner:
            calls = 0
            def train(self, examples, parent_version):
                self.calls += 1
                return Artifact('v1', 'fixture', 'test-only', parent_version, {})
            def prepare(self, artifact): pass
            def retain(self, versions): pass
        learner = self.train(selected, OutdatedLearner())
        self.assertEqual(learner.calls, 0)
        self.assertEqual(self.store.model_version, 'base')
        self.assertIn(selected, self.store.memories())
        job = json.loads(self.store.db.execute("SELECT payload FROM jobs WHERE id='context_probe'").fetchone()[0])
        self.assertEqual(job['state'], 'FAILED')
        self.assertIn('histories', job['reason'])

    def test_cold_selection_skips_nonfitting_and_keeps_filling_in_cold_order(self):
        def memory(name, words, age):
            return Memory(name, 'x '*words, 'y', (Support(name,0),), recency_anchor=age)
        a = memory('a', 3, 0)  # 4 tokens
        b = memory('b', 3, 1)  # Does not fit the remaining 2 tokens.
        c = memory('c', 1, 1)
        oversized = memory('oversized', 20, -1)
        warm = memory('warm', 1, 2)
        blocked = memory('blocked', 1, 0)
        config = replace(self.store.config, region_token_cap=6)
        selected = cold_region((warm, c, oversized, b, blocked, a), 2, self.store.counter,
                               config, excluded=(blocked.ref,))
        self.assertEqual(selected, (a, c))
        self.assertEqual(sum(self.store.counter.count(m.payload) for m in selected), 6)

    def test_original_probe_h_includes_later_long_entry_and_obeys_actual_half_budget(self):
        for name in ('a', 'b', 'z'):
            self.add(name, id=name, content='x' if name!='z' else 'long '*80, applicability='y')
        for i in range(4):
            self.store.record_result(episode('clock'+str(i)), keep_source=False)
        class Echo:
            def __init__(self): self.calls = []
            def consolidate(self, region, mode, batch, budget):
                self.calls.append(tuple(m.id for m in region))
                return tuple(replace(m, id='out_'+m.id, parents=(m.ref,),
                                     consolidated=True, examples=()) for m in region)
        self.store.mark_regular()
        consolidator, learner, events = Echo(), RecordingLearner(), []
        self.runtime = Runtime(self.store, ToyWriter(), consolidator, self.validator, learner, events.append)
        self.runtime.drain()
        self.runtime.capacity_maintenance()
        self.assertEqual(consolidator.calls, [('a','b','z')])
        committed, = [event for event in events if event['event']=='capacity_committed']
        self.assertEqual(committed['budget'], 85//2)
        self.assertLessEqual(self.store.stats()['active_tokens'], 85//2)
        self.runtime.capacity_maintenance()
        self.assertEqual(consolidator.calls[1:], [('out_a','out_b')])
        self.assertEqual(learner.calls, [])
        self.assertEqual(self.store.model_version, 'base')

    def test_failed_compression_keeps_original_region(self):
        for name in ('a','b','z'):
            self.add(name, id=name, content='x' if name!='z' else 'long '*80, applicability='y')
        for i in range(4):
            self.store.record_result(episode('clock'+str(i)), keep_source=False)
        region = self.store.memories()
        class NoCompression:
            def consolidate(self, region, mode, batch, budget):
                return (Memory('too_large', 'x '*90, 'y',
                               tuple(s for m in region for s in m.supports),
                               parents=tuple(m.ref for m in region), consolidated=True),)
        self.store.mark_regular()
        self.runtime = Runtime(self.store, ToyWriter(), NoCompression(), self.validator)
        self.runtime.drain()
        self.runtime.capacity_maintenance()
        self.assertEqual(self.store.memories(), region)

    def test_removed_cold_fraction_is_rejected_in_old_configuration(self):
        path = Path(self.temp.name)/'old_config.json'
        path.write_text(json.dumps({'cold_fraction':0.2}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Unknown configuration fields'):
            Config.load(path)

"""Native-context preservation and recovery of individual consolidation samples."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from memturn.config import Config
from memturn.contracts import Artifact, ExposedMemory, RetrievalEvent, Support, dumps
from memturn.learning.materials import bind_example, rule_question
from memturn.memory.store import Store
from memturn.memory.validation import Validator
from memturn.models.agents import ChatConsolidator, decode_memory
from memturn.runtime.service import Runtime
from memturn.tokens import OfflineCounter
from tests.fixtures.operators import ToyWriter
from tests.test_core import episode


class MaterialRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name)/'state.sqlite', Config(), OfflineCounter())
        self.validator = Validator(self.store)
        for name in ('a', 'b'):
            ep = episode(name)
            self.store.record_result(ep)
            self.store.append(ToyWriter().write(ep, 0))
            self.store.finish_source(name)
        self.value = {'local_id':'S', 'content':'Reusable rule', 'applicability':'When appropriate',
                      'parents':[m.ref for m in self.store.memories()],
                      'supports':[{'task_id':'a','step':0}, {'task_id':'b','step':0}],
                      'examples':[{'example_type':'rule_qa','question':'What strategy applies?'},
                                  {'task_id':'a','step':0}]}
        self.events = []
        self.runtime = None

    def tearDown(self):
        if self.runtime:
            self.runtime.close()
        self.store.close()
        self.temp.cleanup()

    def decode(self, value=None):
        return decode_memory(value or self.value, 0, memory_id='summary', source_lookup=self.store.episode)

    def propose(self, outputs=None, consolidator=None):
        class Operator:
            def consolidate(self, *args):
                return outputs
        self.runtime = Runtime(self.store, None, consolidator or Operator(), self.validator,
                               audit=self.events.append)
        return self.runtime._propose(self.store.memories(), 'regular')

    def test_original_probe_a_keeps_good_pair_after_bad_example(self):
        good = self.decode()
        broken = dict(self.value, examples=[*self.value['examples'], {'example_type':'unsupported'}])
        with self.assertLogs('memturn.models.agents', level='WARNING') as logs:
            recovered = self.decode(broken)
        self.assertEqual(recovered.examples, good.examples)
        self.assertTrue(self.validator.trainable(recovered))
        self.assertIn('unknown example type', logs.output[0])

    def test_decoder_skips_bad_item_between_valid_items(self):
        good = self.decode()
        for bad in (None, {'task_id':[],'step':0}, {'task_id':'missing','step':0}, {'task_id':'a','step':99},
                    {'task_id':'a','step':True}, {'task_id':'a','step':0,'target':'forged'},
                    {'task_id':'a','step':0,'memory_local_id':'other'},
                    {'task_id':'a','step':0,'covered_content_refs':['foreign#content']}):
            with self.subTest(bad=bad), self.assertLogs('memturn.models.agents', level='WARNING'):
                recovered = self.decode(dict(self.value, examples=[self.value['examples'][0], bad,
                                                                    self.value['examples'][1]]))
            self.assertEqual(recovered.examples, good.examples)

    def test_runtime_filters_invalid_samples_without_discarding_valid_pair(self):
        good = self.decode()
        qa, decision = good.examples
        bad = (replace(qa, prompt=None), replace(qa, target='invented'),
               replace(decision, support_refs=(Support('b',0),)),
               replace(decision, covered_content_refs=(123,)), replace(decision, target='invented'))
        recovered, = self.propose((replace(good, examples=(qa, *bad, decision)),))
        self.assertEqual(recovered.examples, good.examples)
        self.assertTrue(self.validator.trainable(recovered))
        event, = self.events
        self.assertEqual((event['invalid'], event['over_budget']), (len(bad), 0))

    def test_incomplete_pair_remains_external_with_its_valid_material(self):
        good = self.decode()
        partial = replace(good, examples=(good.examples[0], replace(good.examples[1], target='forged')))
        recovered, = self.propose((partial,))
        self.assertEqual(recovered.examples, good.examples[:1])
        self.assertFalse(self.validator.trainable(recovered))
        self.store.replace_region(self.store.memories(), (recovered,), 'regular')
        self.assertEqual(self.store.memories(), (recovered,))

    def test_shared_sample_cap_keeps_prefix_and_external_summaries(self):
        self.store.config = replace(self.store.config, max_examples_per_job=2)
        good = self.decode()
        extra = replace(good, examples=(*good.examples, good.examples[1]))
        other = self.decode(dict(self.value, content='Another rule'))
        other = replace(other, id='other', examples=())
        other = replace(other, examples=(rule_question(other), bind_example(good.examples[1], other)))
        first, second = self.propose((extra, other))
        self.assertEqual(first.examples, good.examples)
        self.assertTrue(self.validator.trainable(first))
        self.assertEqual(second.content, other.content)
        self.assertEqual(second.examples, ())
        self.assertEqual(sum(e['over_budget'] for e in self.events), 3)

    def test_chat_consolidator_uses_same_runtime_sample_cap(self):
        self.store.config = replace(self.store.config, max_examples_per_job=2)
        value = dict(self.value, examples=[*self.value['examples'], self.value['examples'][1]])
        replies = iter(({'op':'emit','memory':value}, {'op':'finish'}))
        class Client:
            def complete(self, messages):
                return json.dumps(next(replies))
        recovered, = self.propose(consolidator=ChatConsolidator(Client(), self.store, self.store.counter))
        self.assertEqual(len(recovered.examples), 2)
        self.assertTrue(self.validator.trainable(recovered))

    def test_original_probe_e_keeps_native_observation_and_history_without_retrievals(self):
        selected = self.decode()
        raw = self.store.memories()[0]
        alias = replace(raw, id='alias', content='phase')
        self.store.append(alias)
        ep = self.store.episode('a')
        exposed = ExposedMemory(raw.ref, 'EXTERNAL MEMORY BODY', ('a',))
        ep = replace(ep, retrievals=(RetrievalEvent(0, None, (exposed,), 0),))
        self.store.db.execute('UPDATE episodes SET payload=? WHERE id=?', (dumps(ep.to_dict()), 'a'))
        self.store.db.commit()
        self.runtime = Runtime(self.store, None, None, None)
        sample = bind_example(raw.examples[0], selected)
        self.assertIn('phase', sample.prompt)
        self.assertEqual(self.runtime.training_histories(selected, (sample,)), {('a',0): ()})
        later = replace(sample, step=1, prompt=ep.steps[1].observation, target=ep.steps[1].action)
        histories = self.runtime.training_histories(selected, (later,))
        self.assertEqual(histories[('a',1)], ep.steps[:1])
        self.assertNotIn('EXTERNAL MEMORY BODY', repr(histories))
        self.assertEqual(self.runtime.training_memories(selected), ())

    def test_qa_question_still_cannot_quote_its_own_answer(self):
        good = self.decode()
        qa = replace(good.examples[0], prompt='Explain ' + good.content)
        self.assertFalse(self.validator.valid_example(qa, good))

    def test_missing_history_source_still_fails(self):
        good = self.decode()
        self.runtime = Runtime(self.store, None, None, None)
        with self.assertRaisesRegex(ValueError, 'source is no longer available'):
            self.runtime.training_histories(good, (replace(good.examples[1], task_id='missing'),))

    def test_recovered_material_reaches_context_learner_and_can_publish(self):
        value = dict(self.value, examples=[self.value['examples'][0], {'example_type':'unsupported'},
                                           {'task_id':'a','step':1}],
                     supports=[{'task_id':'a','step':1}, {'task_id':'b','step':0}])
        with self.assertLogs('memturn.models.agents', level='WARNING'):
            recovered, = self.propose((self.decode(value),))
        self.store.replace_region(self.store.memories(), (recovered,), 'regular')
        self.store.append(replace(recovered, id='alias', content='phase', examples=()))
        calls = []
        class Learner:
            accepts_memory_context = True
            def train(self, examples, parent_version, *, memory_context, histories):
                calls.append((examples, memory_context, histories))
                return Artifact('trained', 'fixture', 'test-only', parent_version, {})
            def prepare(self, artifact): pass
            def retain(self, versions): pass
        self.runtime.learner = Learner()
        self.runtime._train('recovery', recovered)
        self.assertEqual(len(calls), 1)
        examples, memories, histories = calls[0]
        self.assertEqual(examples, recovered.examples)
        self.assertEqual(memories, ())
        self.assertEqual(histories[('a',1)], self.store.episode('a').steps[:1])
        self.assertEqual(self.store.model_version, 'trained')
        self.assertNotIn(recovered.ref, {m.ref for m in self.store.memories()})

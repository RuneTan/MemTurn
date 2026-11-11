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

    def test_shared_sample_cap_keeps_required_pair_and_external_summaries(self):
        self.store.config = replace(self.store.config, max_examples_per_job=2)
        good = self.decode()
        ep = self.store.episode('b')
        extra_decision = replace(good.examples[1], task_id='b', prompt=ep.steps[0].observation,
                                 target=ep.steps[0].action, support_refs=(Support('b',0),))
        extra = replace(good, examples=(*good.examples, extra_decision))
        other = self.decode(dict(self.value, content='Another rule'))
        other = replace(other, id='other', examples=())
        other = replace(other, examples=(rule_question(other), bind_example(good.examples[1], other)))
        first, second = self.propose((extra, other))
        self.assertEqual(first.examples, good.examples)
        self.assertTrue(self.validator.trainable(first))
        self.assertEqual(second.content, other.content)
        self.assertEqual(second.examples, ())
        self.assertEqual(sum(e['over_budget'] for e in self.events), 3)

    def _budgeted_material_reaches_learner(self, mode, examples, cap, expected_types):
        self.store.config = replace(self.store.config, training_material=mode, max_examples_per_job=cap)
        memory = self.decode(dict(self.value, examples=examples))
        calls = []
        class Operator:
            def consolidate(self, *args): return (memory,)
        class Learner:
            def train(self, examples, parent_version, *, histories, memory_context=()):
                calls.append(examples)
                return Artifact('budgeted', 'fixture', 'test-only', parent_version, {})
            def prepare(self, artifact): pass
            def retain(self, versions): pass
        self.runtime = Runtime(self.store, None, Operator(), self.validator, Learner(),
                               audit=self.events.append)
        self.runtime.regular_maintenance()
        self.assertEqual(len(calls), 1)
        self.assertEqual([e.example_type for e in calls[0]], expected_types)
        self.assertLessEqual(len(calls[0]), cap)
        self.assertEqual(self.store.model_version, 'budgeted')
        self.assertNotIn(memory.ref, {m.ref for m in self.store.memories()})

    def test_original_mixed_budget_probe_reaches_training(self):
        qa, decision = self.value['examples']
        second_qa = dict(qa, question='When does the strategy apply?')
        self._budgeted_material_reaches_learner('mixed', [qa, second_qa, decision], 2,
                                               ['rule_qa', 'decision'])

    def test_original_qa_only_budget_probe_reaches_training(self):
        qa, decision = self.value['examples']
        self._budgeted_material_reaches_learner('qa_only', [decision, qa], 1, ['rule_qa'])
        filtered = next(e for e in self.events if e['event'] == 'training_material_filtered')
        self.assertEqual((filtered['unused_type'], filtered['over_budget']), (1, 0))

    def test_decision_only_discards_questions_before_applying_budget(self):
        qa, decision = self.value['examples']
        self._budgeted_material_reaches_learner('decision_only', [qa, decision], 1, ['decision'])

    def _two_summaries(self):
        first = self.decode()
        second = replace(first, id='second', content='Another rule', examples=())
        second = replace(second, examples=(rule_question(second), bind_example(first.examples[1], second)))
        extra = replace(first.examples[0], prompt='When does this rule apply?')
        first = replace(first, examples=(first.examples[0], extra, first.examples[1]))
        return first, second

    def test_shared_budget_reserves_required_types_before_extra_samples(self):
        self.store.config = replace(self.store.config, max_examples_per_job=4)
        first, second = self.propose(self._two_summaries())
        self.assertEqual([len(m.examples) for m in (first, second)], [2, 2])
        self.assertTrue(all(self.validator.trainable(m) for m in (first, second)))
        self.assertEqual(sum(e['over_budget'] for e in self.events), 1)

    def test_spare_budget_keeps_extra_questions_and_original_order(self):
        self.store.config = replace(self.store.config, max_examples_per_job=5)
        originals = self._two_summaries()
        kept = self.propose(originals)
        self.assertEqual([m.examples for m in kept], [m.examples for m in originals])
        self.assertEqual(self.events, [])

    def test_incomplete_material_does_not_displace_a_later_complete_pair(self):
        self.store.config = replace(self.store.config, max_examples_per_job=2)
        first, second = self._two_summaries()
        first = replace(first, examples=first.examples[:2])
        kept_first, kept_second = self.propose((first, second))
        self.assertEqual(kept_first.content, first.content)
        self.assertEqual(kept_first.examples, ())
        self.assertTrue(self.validator.trainable(kept_second))

    def test_budget_below_required_pair_preserves_partial_external_material(self):
        self.store.config = replace(self.store.config, max_examples_per_job=1)
        kept, = self.propose((self.decode(),))
        self.assertEqual(len(kept.examples), 1)
        self.assertFalse(self.validator.trainable(kept))
        self.store.replace_region(self.store.memories(), (kept,), 'regular')
        self.assertEqual(self.store.memories(), (kept,))

    def test_chat_consolidator_uses_same_runtime_sample_cap(self):
        self.store.config = replace(self.store.config, max_examples_per_job=2)
        value = dict(self.value, examples=[*self.value['examples'], {'task_id':'b','step':0}])
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
        self.assertEqual(self.runtime.training_histories((sample,)), {('a',0): ()})
        later = replace(sample, step=1, prompt=ep.steps[1].observation, target=ep.steps[1].action)
        histories = self.runtime.training_histories((later,))
        self.assertEqual(histories[('a',1)], ep.steps[:1])
        self.assertNotIn('EXTERNAL MEMORY BODY', repr(histories))
        self.assertEqual(self.runtime.training_memories(selected), ())

    def test_qa_text_overlap_is_audited_instead_of_rejected(self):
        good = self.decode()
        qa = replace(good.examples[0], prompt='Explain ' + good.content)
        self.assertTrue(self.validator.valid_example(qa, good))
        recovered, = self.propose((replace(good, examples=(qa,good.examples[1])),))
        self.assertTrue(self.validator.trainable(recovered))
        self.assertEqual(self.events[0]['event'], 'qa_text_overlap')
        self.assertEqual(self.events[0]['count'], 1)

    def test_missing_history_source_still_fails(self):
        good = self.decode()
        self.runtime = Runtime(self.store, None, None, None)
        with self.assertRaisesRegex(ValueError, 'source is no longer available'):
            self.runtime.training_histories((replace(good.examples[1], task_id='missing'),))

    def test_qa_answer_inside_another_word_does_not_block_training(self):
        value = dict(self.value, content='open', examples=[
            {'example_type':'rule_qa','question':'How do I reopen the door?'},
            self.value['examples'][1]])
        recovered, = self.propose((self.decode(value),))
        self.assertTrue(self.validator.trainable(recovered))
        self.assertEqual(self.events[0]['event'], 'qa_text_overlap')

    def test_duplicate_references_do_not_discard_a_good_sample(self):
        qa = dict(self.value['examples'][0], covered_content_refs=['content','S#content','content'])
        memory = self.decode(dict(self.value, examples=[qa,self.value['examples'][1]]))
        self.assertEqual(len(memory.examples), 2)
        self.assertEqual(memory.examples[0].covered_content_refs, (memory.ref+'#content',))
        self.assertTrue(self.validator.trainable(memory))
        decision = replace(memory.examples[1], support_refs=(Support('a',0), Support('a',0)))
        self.assertTrue(self.validator.valid_example(decision, memory))
        self.assertFalse(self.validator.valid_example(replace(decision,
            support_refs=(Support('a',0), Support('b',0))), memory))
        # Persisted repeated refs are also harmless; foreign refs remain invalid.
        duplicated = replace(memory.examples[0], covered_content_refs=(memory.ref+'#content',)*2)
        self.assertTrue(self.validator.valid_example(duplicated, memory))
        self.assertFalse(self.validator.valid_example(replace(duplicated,
            covered_content_refs=(*duplicated.covered_content_refs,'foreign@1#content')), memory))

    def test_identical_samples_do_not_consume_budget_twice(self):
        self.store.config = replace(self.store.config, max_examples_per_job=4)
        good = self.decode()
        repeated = replace(good, examples=good.examples*2)
        other = replace(good, id='other', content='Another rule', examples=())
        other = replace(other, examples=(rule_question(other),bind_example(good.examples[1],other)))
        first, second = self.propose((repeated, other))
        self.assertEqual(first.examples, good.examples)
        self.assertEqual(second.examples, other.examples)
        self.assertEqual(self.validator.training_examples(repeated), good.examples)
        self.assertTrue(self.validator.trainable(repeated))
        self.assertEqual(self.events[0]['duplicates'], 2)
        self.assertEqual(self.events[0]['over_budget'], 0)

    def _multiple_questions_reach_learner(self, mode):
        self.store.config = replace(self.store.config, training_material=mode)
        second_qa = {'example_type':'rule_qa','question':'Under which conditions does this rule apply?'}
        value = dict(self.value, examples=[self.value['examples'][0], second_qa,
                                          self.value['examples'][0], self.value['examples'][1],
                                          {'task_id':'b','step':0}])
        memory = self.decode(value)
        calls = []
        class Operator:
            def consolidate(self, *args): return (memory,)
        class Learner:
            def train(self, examples, parent_version, *, histories, memory_context=()):
                calls.append((examples, histories, memory_context))
                return Artifact('multi_qa', 'fixture', 'test-only', parent_version, {})
            def prepare(self, artifact): pass
            def retain(self, versions): pass
        self.runtime = Runtime(self.store, None, Operator(), self.validator, Learner(),
                               audit=self.events.append)
        self.runtime.regular_maintenance()
        self.assertEqual(len(calls), 1)
        examples, histories, memories = calls[0]
        questions = [e.prompt for e in examples if e.example_type == 'rule_qa']
        self.assertEqual(questions, ['What strategy applies?', second_qa['question']])
        decisions = [e for e in examples if e.example_type == 'decision']
        self.assertEqual(len(decisions), 2 if mode == 'mixed' else 0)
        expected_histories = {('a',0): (), ('b',0): ()} if mode == 'mixed' else {}
        self.assertEqual(histories, expected_histories)
        self.assertEqual(memories, ())
        self.assertEqual(self.store.model_version, 'multi_qa')
        self.assertNotIn(memory.ref, {m.ref for m in self.store.memories()})
        filtered = next(e for e in self.events if e['event'] == 'training_material_filtered')
        self.assertEqual(filtered['duplicates'], 1)

    def test_multiple_valid_questions_train_and_publish_in_mixed_mode(self):
        self._multiple_questions_reach_learner('mixed')

    def test_multiple_valid_questions_train_and_publish_in_qa_only_mode(self):
        self._multiple_questions_reach_learner('qa_only')

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

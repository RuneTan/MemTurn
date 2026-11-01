"""Admission checks provenance and fidelity, not summary semantics."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from memturn.config import Config
from memturn.contracts import Support, dumps
from memturn.memory.store import Store
from memturn.memory.validation import Validator
from memturn.models.agents import decode_memory
from memturn.tokens import OfflineCounter
from tests.fixtures.operators import ToyWriter
from tests.test_core import episode


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name)/'state.sqlite', Config(), OfflineCounter())
        self.ep = episode('a')
        self.store.record_result(self.ep)
        self.raw = ToyWriter().write(self.ep, 0)
        self.store.append(self.raw)
        self.validator = Validator(self.store)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_free_form_paraphrase_and_unproven_claim_are_not_semantically_judged(self):
        for text in ('Inspect the lock before proceeding.', 'An unproven general claim.'):
            self.assertTrue(self.validator.valid_memory(replace(self.raw,
                content=text, applicability='A freely worded condition')))

    def test_invalid_task_step_and_boolean_references_rejected(self):
        for ref in (Support('missing', 0), Support('a', -1), Support('a', 999),
                    Support('a', True), Support('a', 0.0), Support('', 0), Support([], 0)):
            self.assertFalse(self.validator.valid_memory(replace(self.raw, supports=(ref,))))

    def test_coverage_deduplicates_tasks_and_does_not_require_score_gain(self):
        second = replace(self.ep, task_id='b', reward=0, success=False)
        self.store.record_result(second)
        refs = (Support('a', 0), Support('a', 0), Support('a', 1), Support('b', 0))
        memory = replace(self.raw, supports=refs)
        self.assertTrue(self.validator.valid_memory(memory))
        self.assertEqual(self.validator.source_coverage_count(memory), 2)

    def test_citation_outside_declared_parent_tasks_rejected(self):
        self.store.record_result(replace(self.ep, task_id='b'))
        memory = replace(self.raw, id='summary', parents=(self.raw.ref,),
                         consolidated=True, supports=(Support('b', 0),))
        self.assertFalse(self.validator.validate_replacement((self.raw,), (memory,)))

    def decision_memory(self):
        ref = Support('a', 0)
        example = replace(self.raw.examples[0], memory_ref=self.raw.ref,
            covered_content_refs=(self.raw.ref+'#content',), support_refs=(ref,))
        return replace(self.raw, examples=(example,)), example

    def test_decision_material_must_match_original_observation_and_action(self):
        memory, example = self.decision_memory()
        self.assertTrue(self.validator.valid_example(example, memory))
        for forged in (replace(example, prompt='invented observation'),
                       replace(example, target='invented action'), replace(example, input_step=True)):
            self.assertFalse(self.validator.valid_example(forged, memory))

    def test_unaccepted_action_rejected_without_progress_gate(self):
        memory, example = self.decision_memory()
        changed = replace(self.ep, steps=(replace(self.ep.steps[0], accepted=False), *self.ep.steps[1:]))
        self.store.db.execute('UPDATE episodes SET payload=? WHERE id=?',
            (dumps(changed.to_dict()), 'a'))
        self.store.db.commit()
        self.assertFalse(self.validator.valid_example(example, memory))

    def test_decoder_copies_sources_and_does_not_trust_supplied_targets(self):
        value = {'content':'A summary', 'applicability':'condition',
                 'supports':[{'task_id':'a','step':0}],
                 'examples':[{'task_id':'a','step':0}]}
        memory = decode_memory(value, 0, source_lookup=self.store.episode)
        self.assertEqual(memory.examples[0].prompt, self.ep.steps[0].observation)
        self.assertEqual(memory.examples[0].target, self.ep.steps[0].action)
        value['examples'][0]['target'] = 'fabricated'
        with self.assertRaisesRegex(ValueError, 'differs'):
            decode_memory(value, 0, source_lookup=self.store.episode)

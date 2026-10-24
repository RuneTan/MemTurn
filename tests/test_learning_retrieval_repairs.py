import json
import importlib.util
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from memturn.config import Config
from memturn.contracts import Example, Memory, RetrievalResult, Support, Task
from memturn.learning.huggingface import HuggingFaceLearner, SFTConfig, epoch_schedule, training_schedule
from memturn.memory.consolidation import ToyConsolidator, ToyWriter
from memturn.memory.retrieval import Retriever
from memturn.memory.store import Store
from memturn.memory.validation import Validator
from memturn.models.http import task_messages
from memturn.runtime.service import Runtime
from memturn.tasks.toy import ToyEnvironment, ToyPolicy, ToyVerifier
from memturn.tokens import OfflineCounter


class LearningRetrievalRepairs(unittest.TestCase):
    def test_epoch_mode_covers_all_examples_with_one_update_per_example(self):
        for size in (2, 4, 9):
            schedule = epoch_schedule(size, 1, 1, 42)
            self.assertEqual(len(schedule), size)
            self.assertTrue(all(len(batch) == 1 for batch in schedule))
            self.assertEqual(sorted(sum(schedule, ())), list(range(size)))

    def test_epoch_mode_does_not_drop_the_last_partial_microbatch(self):
        schedule = epoch_schedule(5, 2, 2, 42)
        self.assertEqual([len(batch) for batch in schedule], [2, 2, 1, 2, 2, 1])
        self.assertEqual(sorted(sum(schedule[:3], ())), list(range(5)))
        self.assertEqual(sorted(sum(schedule[3:], ())), list(range(5)))

    def test_small_optimizer_budget_still_covers_every_sample(self):
        schedule = training_schedule(4, 2, 1, 42)
        self.assertEqual(len(schedule), 2)
        self.assertEqual(len(schedule[0]), 2)
        self.assertEqual(set(sum(schedule, ())), set(range(4)))
        self.assertEqual(schedule, training_schedule(4, 2, 1, 42))

    def test_training_schedule_covers_before_repeating(self):
        order = sum(training_schedule(5, 4, 2, 1), ())
        self.assertEqual(set(order[:5]), set(range(5)))

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                         'optional language-model dependencies not installed')
    def test_insufficient_token_budget_rejects_entire_job_before_model_load(self):
        class Tokenizer:
            def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
                return ''.join(f"<{m['role']}>{m['content']}" for m in messages) + ('<assistant>' if add_generation_prompt else '!')
            def encode(self, text, **kwargs): return list(text.encode())
        with tempfile.TemporaryDirectory() as folder:
            learner = HuggingFaceLearner(folder, SFTConfig('unused', 'fixed', ('q_proj',),
                loader_class='AutoModelForCausalLM', max_training_tokens=1000), None)
            examples = tuple(Example(str(i), 0, 'public observation', 'inspect_lock') for i in range(3))
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=Tokenizer()), \
                 patch('transformers.AutoModelForCausalLM.from_pretrained') as load:
                with self.assertRaisesRegex(ValueError, 'cannot cover all'):
                    learner.train(examples, 'base')
                load.assert_not_called()

    def test_scope_rejects_wrong_versions_but_is_optional(self):
        counter, config = OfflineCounter(), Config()
        memories = (Memory('a', 'refresh_token', 'api:v1', (Support('a', 0),)),
                    Memory('b', 'renew_session', 'api:v2', (Support('b', 0),)),
                    Memory('c', 'inspect_lock', 'vault:v1', (Support('c', 0),)))
        retriever = Retriever(counter, config)
        query = '{"family":"api","version":"v2"}'
        self.assertGreater(len(retriever.search(memories, query)), 1)
        self.assertEqual(retriever.search(memories, query, applicability='api:v2'), (memories[1],))
        self.assertEqual(retriever.search(memories[:1], query, applicability='api:v2'), ())

    def test_search_call_result_and_availability_are_visible(self):
        memory = Memory('a', 'a rule', 'api:v1', (Support('a', 0),))
        messages = task_messages('observation', (memory,),
            (RetrievalResult(None, (), 1), RetrievalResult('api version 1', (memory,), 0)))
        self.assertNotIn('"kind":"memory_search"', messages[0]['content'])
        self.assertIn('unavailable', messages[0]['content'])
        self.assertEqual(json.loads(messages[2]['content']),
                         {'kind': 'memory_search', 'query': 'api version 1'})
        result = json.loads(messages[3]['content'])['memory_search_result']
        self.assertEqual(result['remaining'], 0)
        self.assertEqual(result['results'][0]['ref'], memory.ref)

    def test_policy_prompt_has_no_copyable_command_or_query_placeholders(self):
        messages = task_messages('observation', (), (RetrievalResult(None, (), 2),))
        self.assertNotIn('one environment command', messages[0]['content'])
        self.assertNotIn('specific query', messages[0]['content'])
        self.assertIn('actual environment command', messages[0]['content'])

    def test_runtime_records_search_without_fabricating_environment_actions(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / 'state.sqlite', Config(), OfflineCounter())
            runtime = Runtime(store, ToyWriter(), ToyConsolidator(store), Validator(store, ToyVerifier()))
            class Policy:
                def __init__(self): self.calls, self.histories = 0, []
                def act(self, observation, memories, version, history):
                    self.histories.append(history)
                    self.calls += 1
                    if self.calls <= 3:
                        return {'kind': 'memory_search', 'query': 'vault v1'}
                    return 'inspect_lock' if self.calls == 4 else 'open'
            policy = Policy()
            try:
                result = runtime.run_task(Task('probe', 'vault', 'v1', metadata={'evaluation_only':True}),
                                          ToyEnvironment(), policy)
                self.assertTrue(result.success)
                self.assertEqual([s.action for s in result.steps], ['inspect_lock', 'open'])
                self.assertEqual([s for s in policy.histories[-1] if isinstance(s, RetrievalResult)][-1].error,
                                 'search_budget_exhausted')
                self.assertEqual(store.stats()['completed'], 0)
            finally:
                runtime.close()
                store.close()

    def test_disabled_memory_is_not_leaked_through_tool_history(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / 'state.sqlite', Config(), OfflineCounter())
            runtime = Runtime(store, ToyWriter(), ToyConsolidator(store), Validator(store, ToyVerifier()))
            class Policy:
                def act(self, observation, memories, version, history):
                    self.memories, self.history = memories, history
                    return 'inspect_lock' if json.loads(observation)['phase'] == 'prepare' else 'open'
            policy = Policy()
            try:
                runtime.run_task(Task('probe', 'vault', 'v1', metadata={'evaluation_only':True, 'memory_enabled':False}),
                                 ToyEnvironment(), policy)
                self.assertEqual(policy.memories, ())
                self.assertEqual(policy.history[0], RetrievalResult(None, (), 0))
            finally:
                runtime.close()
                store.close()

    def test_replay_projection_removes_feedback_only_after_environment_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / 'state.sqlite', Config(), OfflineCounter())
            runtime = Runtime(store, ToyWriter(), ToyConsolidator(store), Validator(store, ToyVerifier()))
            class Policy:
                def act(self, observation, memories, version, history):
                    state = json.loads(observation)
                    return 'open' if not state['feedback'] or state['phase'] == 'finish' else 'inspect_lock'
            try:
                source = runtime.run_task(Task('source', 'vault', 'v1'), ToyEnvironment(), Policy())
                runtime.drain()
                memory = ToyConsolidator(store).consolidate(store.memories(), 'regular', 0, 2048)[0]
                example = memory.examples[0]
                self.assertEqual(example.step, 1)
                self.assertEqual(example.input_step, 0)
                self.assertEqual(json.loads(example.prompt)['feedback'], '')
                self.assertTrue(ToyVerifier().verify_example(example, source))
                self.assertFalse(ToyVerifier().verify_example(replace(example, target='renew_session'), source))
                self.assertFalse(ToyVerifier().verify_example(replace(example, input_step=2), source))
                changed = json.loads(example.prompt); changed['object'] += 1
                self.assertFalse(ToyVerifier().verify_example(replace(example, prompt=json.dumps(changed)), source))
            finally:
                runtime.close()
                store.close()

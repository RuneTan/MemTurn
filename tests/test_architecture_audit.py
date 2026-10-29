"""Regression cases derived from the October 24 architecture audit."""
from dataclasses import replace
import json
import hashlib
import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from memturn.config import Config
from memturn.contracts import Artifact, Memory, Step, Support, Task
from tests.fixtures.operators import ToyConsolidator, ToyWriter
from memturn.memory.policies import cold_region
from memturn.memory.store import BudgetExceeded, StaleSnapshot, Store
from memturn.memory.validation import Validator
from memturn.models.agents import ChatWriter
from memturn.runtime.service import Runtime
from tests.fixtures.environment import ToyEnvironment, ToyPolicy, ToyVerifier
from memturn.tokens import OfflineCounter
from tests.test_core import episode


class ArchitectureAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name)/'state.sqlite', Config(), OfflineCounter())
        self.validator = Validator(self.store, ToyVerifier())
        self.runtime = None

    def tearDown(self):
        if self.runtime:
            self.runtime.close()
        self.store.close()
        self.temp.cleanup()

    def add(self, name):
        ep = episode(name)
        self.store.record_result(ep)
        raw = ToyWriter().write(ep, self.store.batch)
        self.store.append(raw)
        self.store.finish_source(name)
        return raw

    def runtime_for(self, writer=None, consolidator=None, learner=None):
        self.runtime = Runtime(self.store, writer or ToyWriter(),
                               consolidator or ToyConsolidator(self.store), self.validator, learner)
        return self.runtime

    def summary(self):
        return ToyConsolidator(self.store).consolidate(self.store.memories(), 'regular', self.store.batch, 2048)[0]

    def test_search_does_not_spend_environment_action_budget(self):
        class Policy:
            calls = 0
            def act(self, *args):
                self.calls += 1
                return ({'kind':'memory_search','query':'vault'} if self.calls == 1 else
                        'inspect_lock' if self.calls == 2 else 'open')
        result = self.runtime_for().run_task(Task('two_actions','vault','v1',max_steps=2),
                                              ToyEnvironment(),Policy())
        self.assertTrue(result.success)
        self.assertEqual(len(result.steps),2)

    def test_same_batch_reaccess_invalidates_cold_snapshot(self):
        self.add('a')
        for i in range(4):
            self.store.record_result(episode(f'clock{i}'), keep_source=False)
        self.store.touch(self.store.memories())
        region = self.store.memories()
        outputs = (self.summary(),)
        self.store.touch(region)
        self.assertEqual(region[0].recency_anchor,self.store.memories()[0].recency_anchor)
        with self.assertRaises(StaleSnapshot):
            self.store.replace_region(region,outputs,'capacity')

    def test_same_revision_rewrite_is_rejected(self):
        raw = self.add('a')
        with self.assertRaises(ValueError):
            self.store.replace_region((raw,),(replace(raw,content='changed',parents=(raw.ref,)),),'regular')
        self.assertEqual(self.store.memories()[0],raw)

    def test_publication_record_failure_rolls_back_pointer_and_exit(self):
        raw = self.add('a')
        with patch.object(self.store,'_job',side_effect=BudgetExceeded('injected ledger budget')):
            with self.assertRaises(BudgetExceeded):
                self.store.publish(raw,Artifact('v1','fixture','test','base',{}),job_id='publication')
        self.assertEqual(self.store.model_version,'base')
        self.assertEqual(self.store.memories(),(raw,))
        self.store.publish(raw,Artifact('v1','fixture','test','base',{}),job_id='publication')
        self.store.recover_jobs()
        row = self.store.db.execute("SELECT payload FROM jobs WHERE id='publication'").fetchone()[0]
        self.assertEqual(json.loads(row)['state'],'COMMITTED')

    def test_external_commit_and_job_record_are_atomic(self):
        raw = self.add('a')
        with patch.object(self.store,'_job',side_effect=BudgetExceeded('ledger full')):
            with self.assertRaises(BudgetExceeded):
                self.store.replace_region((raw,),(self.summary(),),'regular',job_id='regular')
        self.assertEqual(self.store.memories(),(raw,))

    def test_result_and_pending_source_rollback_together_on_failure(self):
        original = self.store._check
        def broken():
            if self.store.db.execute('SELECT count(*) FROM episodes').fetchone()[0]:
                raise RuntimeError('crash before commit')
            original()
        with patch.object(self.store,'_check',side_effect=broken):
            with self.assertRaises(RuntimeError): self.store.record_result(episode('crashed'))
        self.assertEqual(self.store.stats()['completed'],0)
        self.assertIsNone(self.store.episode('crashed'))
        self.assertTrue(self.store.record_result(episode('crashed')))

    def test_failed_writer_does_not_freeze_maintenance_clock(self):
        class Broken:
            def write(self,*args): raise ValueError('malformed output')
        runtime = self.runtime_for(writer=Broken())
        for i in range(4):
            runtime.run_task(Task(f'failed{i}','vault','v1'),ToyEnvironment(),ToyPolicy())
            runtime.drain()
        self.assertEqual(self.store._get('regular_batch'),1)
        self.assertEqual(self.store.stats()['refused_writes'],4)

    def test_later_batch_is_not_marked_processed_by_running_job(self):
        runtime = self.runtime_for()
        for i in range(4): self.store.record_result(episode(f'first{i}'),keep_source=False)
        original = runtime.regular_maintenance
        def during(*args,**kwargs):
            original(*args,**kwargs)
            for i in range(4): self.store.record_result(episode(f'later{i}'),keep_source=False)
        with patch.object(runtime,'regular_maintenance',side_effect=during):
            runtime._run_regular_if_due()
        self.assertEqual(self.store._get('regular_batch'),1)
        self.assertTrue(self.store.regular_due())

    def test_oversized_oldest_does_not_starve_usable_cold_memory(self):
        large = Memory('a','large '*5000,'vault:v1',(Support('a',0),))
        small = Memory('b','small','vault:v1',(Support('b',0),))
        self.assertEqual(cold_region((large,small),2,self.store.counter,self.store.config),(small,))

    def test_exact_copy_with_independent_sources_is_excluded(self):
        self.add('a'); self.add('b')
        chosen = self.summary()
        self.store.replace_region(self.store.memories(),(chosen,),'regular')
        alias = replace(chosen,id='independent_copy',supports=(Support('unrelated',0),))
        self.store.append(alias)
        self.assertEqual(self.runtime_for().training_memories(chosen),())

    def test_repeated_rule_in_old_action_is_not_silently_trained(self):
        self.add('a'); self.add('b')
        chosen = self.summary()
        ep = self.store.episode('a')
        tampered = replace(ep,steps=(replace(ep.steps[0],action=chosen.content),*ep.steps[1:]))
        example = replace(chosen.examples[1],step=1,prompt='public later state')
        with patch.object(self.store,'episode',return_value=tampered):
            with self.assertRaisesRegex(ValueError,'repeats excluded memory'):
                self.runtime_for().training_histories(chosen,(example,))

    def test_writer_cannot_claim_another_existing_episode(self):
        self.add('a')
        class Client:
            def complete(self,messages):
                return json.dumps({'memory':{'content':'first call inspect_lock.', 'applicability':'vault:v1',
                    'supports':[{'task_id':'a','step':0}], 'examples':[]}})
        with self.assertRaisesRegex(ValueError,'current episode'):
            ChatWriter(Client()).write(episode('b'),0)

    def test_domain_verifier_exception_fails_closed(self):
        raw = self.add('a')
        class Broken:
            def verify_support(self,*args): raise RuntimeError('unavailable checker')
        validator = Validator(self.store,Broken())
        self.assertFalse(validator.valid_memory(raw))
        self.assertEqual(validator.support_count(raw),0)

    def test_evaluation_runtime_does_not_recover_or_consume_pending_jobs(self):
        self.store.record_result(episode('pending'))
        self.store.job('unfinished',state='READY')
        self.runtime = Runtime(self.store,None,None,None)
        result = self.runtime.run_task(Task('probe','vault','v1',metadata={'evaluation_only':True}),
                                       ToyEnvironment(),ToyPolicy())
        self.runtime.drain()
        self.assertTrue(result.success)
        self.assertEqual(len(self.store.pending_episodes()),1)
        state = json.loads(self.store.db.execute("SELECT payload FROM jobs WHERE id='unfinished'").fetchone()[0])
        self.assertEqual(state['state'],'READY')

    def test_environment_cleanup_failure_does_not_lose_task_result(self):
        class Environment(ToyEnvironment):
            def close(self): raise RuntimeError('close failed')
        result = self.runtime_for().run_task(Task('cleanup','vault','v1'),Environment(),ToyPolicy())
        self.assertTrue(result.success)
        self.assertEqual(self.store.stats()['completed'],1)

    def test_one_rule_question_per_mixed_candidate(self):
        self.add('a'); self.add('b')
        memory = self.summary()
        self.assertTrue(self.validator.trainable(memory))
        self.assertFalse(self.validator.trainable(replace(memory,examples=(*memory.examples,memory.examples[0]))))

    def test_non_model_consolidator_also_obeys_output_budget(self):
        self.add('a')
        memory = replace(self.summary(), content='inspect_lock '+ 'long '*3000)
        class Operator:
            def consolidate(self,*args): return (memory,)
        with self.assertRaises(BudgetExceeded):
            self.runtime_for(consolidator=Operator())._propose(self.store.memories(),'regular')

    def test_failed_new_region_does_not_hide_existing_eligible_candidate(self):
        self.add('a'); self.add('b')
        candidate=self.summary()
        self.store.replace_region(self.store.memories(),(candidate,),'regular')
        raw=self.add('c')
        class Broken:
            def consolidate(self,*args): return ()
        class Learner:
            def train(self,examples,parent): return Artifact('trained','fixture','test',parent,{})
            def prepare(self,*args): pass
            def retain(self,*args): pass
        self.runtime_for(consolidator=Broken(),learner=Learner()).regular_maintenance()
        self.assertEqual(self.store.model_version,'trained')
        self.assertEqual([m.ref for m in self.store.memories()],[raw.ref])

    def test_retired_version_prevents_next_training_job(self):
        self.add('a'); self.add('b')
        memory = self.summary()
        self.store.replace_region(self.store.memories(),(memory,),'regular')
        class Learner:
            calls=0
            def train(self,*args): self.calls+=1; raise AssertionError('must wait for old task')
            def retain(self,*args): pass
        learner = Learner()
        runtime = self.runtime_for(learner=learner)
        with self.store.session():
            # Model publication is simulated without retiring this candidate.
            with self.store.transaction(): self.store._set('model_version','current')
            runtime._train('waiting',memory)
        self.assertEqual(learner.calls,0)

    def test_audit_sink_failure_cannot_turn_committed_task_into_exception(self):
        runtime = self.runtime_for()
        def broken(event): raise OSError('audit disk full')
        runtime.audit = broken
        result = runtime.run_task(Task('audit','vault','v1'),ToyEnvironment(),ToyPolicy())
        runtime.drain()
        self.assertTrue(result.success)
        self.assertTrue(any('audit_failed' in error for error in runtime.errors))

    def test_background_reservation_blocks_overbudget_work_and_releases(self):
        cap = self.store.config.snapshot_byte_cap
        with self.store.reserve_work(cap):
            self.assertEqual(self.store.stats()['work_reserved_bytes'],cap)
            with self.assertRaises(BudgetExceeded):
                with self.store.session(): pass
        self.assertEqual(self.store.stats()['work_reserved_bytes'],0)

    def test_queue_payload_is_charged_in_addition_to_source(self):
        self.store.record_result(episode('pending'))
        stats = self.store.stats()
        self.assertGreater(stats['queued_source_bytes'],0)
        self.assertEqual(stats['source_training_bytes'],2*stats['queued_source_bytes'])



    def test_shutdown_is_idempotent_and_rejects_new_tasks(self):
        runtime = self.runtime_for()
        runtime.close()
        runtime.close()
        with self.assertRaisesRegex(RuntimeError,'closed'):
            runtime.run_task(Task('late','vault','v1'),ToyEnvironment(),ToyPolicy())

    def test_shutdown_waits_for_task_before_stopping_writer(self):
        runtime = self.runtime_for()
        entered, release = threading.Event(), threading.Event()
        class Slow(ToyPolicy):
            def act(self,*args):
                entered.set()
                if not release.wait(3): raise RuntimeError('test timed out')
                return super().act(*args)
        task_thread = threading.Thread(target=lambda:runtime.run_task(Task('inflight','vault','v1'),ToyEnvironment(),Slow()))
        task_thread.start()
        self.assertTrue(entered.wait(2))
        closing = threading.Thread(target=runtime.close)
        closing.start()
        release.set()
        task_thread.join(4); closing.join(4)
        self.assertFalse(closing.is_alive())
        self.assertEqual(self.store.stats()['completed'],1)
        self.assertEqual(len(self.store.memories()),1)

    def test_slow_adapter_collection_does_not_hold_database_lock(self):
        entered, release = threading.Event(), threading.Event()
        class Learner:
            def retain(self,versions):
                entered.set()
                if not release.wait(3): raise RuntimeError('test timed out')
        runtime = self.runtime_for(learner=Learner())
        collector = threading.Thread(target=runtime.collect_adapters)
        collector.start()
        self.assertTrue(entered.wait(2))
        finished = threading.Event()
        reader = threading.Thread(target=lambda:(self.store.stats(),finished.set()))
        reader.start()
        try:
            self.assertTrue(finished.wait(1),'adapter I/O blocked foreground database access')
        finally:
            release.set(); collector.join(4); reader.join(4)

    def test_recovery_does_not_activate_ready_only_candidate(self):
        artifact = Artifact('candidate','fixture','test','base',{})
        with self.store.transaction():
            self.store._set('model_version','candidate')
            self.store._set('artifact',artifact.__dict__)
        self.store.job('uncommitted',state='READY',artifact=artifact.__dict__)
        with self.assertRaisesRegex(ValueError,'committed publication'):
            self.store.recover_jobs()

    def test_current_publication_survives_job_history_rotation(self):
        raw=self.add('a')
        self.store.publish(raw,Artifact('current','fixture','test','base',{}))
        for i in range(self.store.config.max_job_records+1):
            self.store.job(f'later_{i}',state='SKIPPED')
        self.store.recover_jobs()
        self.assertEqual(self.store.model_version,'current')
        self.assertEqual(self.store._get('publication')['state'],'COMMITTED')




if __name__ == '__main__': unittest.main()

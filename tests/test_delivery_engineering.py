"""Recovery, deadlines, service accounting and deployment failure regressions."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from memturn.config import Config
from memturn.contracts import Artifact, Episode, Memory, Step, Support, Task
from memturn.memory.consolidation import ToyConsolidator, ToyWriter
from memturn.memory.store import BudgetExceeded, Store
from memturn.memory.validation import Validator
from memturn.models.budget import TokenBudgetClient
from memturn.models.http import ChatConfig, HttpChat
from memturn.models.preflight import check_deployment
from memturn.models.publication import VLLMAdapters
from memturn.runtime.arrival import run_arrivals, utility_function
from memturn.runtime.process import ProcessRPC
from memturn.runtime.recovery import RunLease, inspect_manifest
from memturn.runtime.service import Runtime
from memturn.tasks.observed import check_transition, positive_progress
from memturn.tasks.toy import ToyEnvironment, ToyPolicy, ToyVerifier
from memturn.tokens import OfflineCounter


class DeliveryEngineeringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = replace(Config(), max_job_records=2, stream_window=4)
        self.store = Store(self.root/'state.sqlite', self.config, OfflineCounter())
        self.runtime = None

    def tearDown(self):
        if self.runtime: self.runtime.close()
        self.store.close()
        self.temp.cleanup()

    def stream(self):
        self.store.configure_stream('test', 'a'*64)

    def task(self, i):
        return Task(str(i), 'vault', 'v1')

    def complete(self, i):
        self.store.claim_task(self.task(i), i)
        self.store.record_result(Episode(str(i),'vault','v1',(),1.0,True), keep_source=False, sequence=i)

    def test_old_completed_positions_are_not_recounted_after_result_rotation(self):
        self.stream()
        for i in range(10): self.complete(i)
        self.assertFalse(self.store.claim_task(self.task(0), 0))
        self.assertEqual(self.store.stats()['completed'], 10)
        self.assertEqual(self.store.stats()['stream_slots'], 0)
        self.assertEqual(self.store.stats()['stream']['high_watermark'], 9)

    def test_out_of_order_completion_and_restart_preserve_gaps(self):
        self.stream()
        self.store.claim_task(self.task(0), 0)
        self.complete(1)
        self.assertEqual(self.store.stats()['stream']['high_watermark'], -1)
        self.store.close()
        self.store = Store(self.root/'state.sqlite', self.config, OfflineCounter())
        self.store.recover_stream()
        self.assertTrue(self.store.claim_task(self.task(0), 0))
        self.assertFalse(self.store.claim_task(self.task(1), 1))
        self.store.record_result(Episode('0','vault','v1',(),1,True), keep_source=False, sequence=0)
        self.assertEqual(self.store.stats()['stream']['high_watermark'], 1)
        self.assertEqual(self.store.stats()['completed'], 2)

    def test_stream_change_and_unbounded_gap_are_rejected(self):
        self.stream()
        with self.assertRaises(ValueError): self.store.configure_stream('test','b'*64)
        with self.assertRaises(BudgetExceeded): self.store.claim_task(self.task(10), 10)
        self.store.claim_task(self.task(1), 1)
        with self.assertRaises(ValueError): self.store.claim_task(Task('other','vault','v1'), 1)

    def test_completion_failure_rolls_back_stream_watermark(self):
        self.stream()
        self.store.claim_task(self.task(0),0)
        with patch.object(self.store,'_check',side_effect=RuntimeError('fault')):
            with self.assertRaises(RuntimeError):
                self.store.record_result(Episode('0','vault','v1',(),1,True),sequence=0)
        self.assertEqual(self.store.stats()['stream']['high_watermark'],-1)
        self.assertEqual(self.store.stats()['completed'],0)

    def test_run_directory_lease_is_exclusive_and_released(self):
        with RunLease(self.root/'run.lock'):
            with self.assertRaises(RuntimeError):
                with RunLease(self.root/'run.lock'): pass
        with RunLease(self.root/'run.lock'): pass

    def test_manifest_hash_ignores_whitespace_but_rejects_duplicate_ids(self):
        path = self.root/'tasks.jsonl'
        path.write_text(json.dumps(self.task(0).__dict__)+'\n',encoding='utf-8')
        digest = inspect_manifest(path)['sha256']
        path.write_text('\n'+json.dumps(self.task(0).__dict__, separators=(',',':'))+'\n',encoding='utf-8')
        self.assertEqual(inspect_manifest(path)['sha256'],digest)
        path.write_text((json.dumps(self.task(0).__dict__)+'\n')*2,encoding='utf-8')
        with self.assertRaises(ValueError): inspect_manifest(path)

    def test_stalled_worker_is_actually_terminated(self):
        rpc = ProcessRPC([sys.executable,'-u','-c','import time; time.sleep(30)'], timeout=0.25)
        started = time.monotonic()
        with self.assertRaises(TimeoutError): rpc.request('hang')
        self.assertIsNotNone(rpc.process.poll())
        self.assertLess(time.monotonic()-started, 7)

    def test_worker_that_never_reads_cannot_block_large_input_forever(self):
        rpc = ProcessRPC([sys.executable,'-u','-c','import time; time.sleep(30)'], timeout=0.25)
        with self.assertRaises(TimeoutError): rpc.request('hang', text='x'*1048576)
        self.assertIsNotNone(rpc.process.poll())

    def test_worker_output_is_bounded(self):
        rpc = ProcessRPC([sys.executable,'-u','-c',"import sys; sys.stdin.readline(); print('x'*10000,flush=True)"],
                         timeout=3, max_bytes=1024)
        with self.assertRaises(RuntimeError): rpc.request('overflow')
        self.assertIsNotNone(rpc.process.poll())

    def test_worker_mismatched_response_is_rejected(self):
        rpc = ProcessRPC([sys.executable,'-u','-c',"import sys; sys.stdin.readline(); print('{\"id\":999,\"result\":1}',flush=True)"], timeout=3)
        with self.assertRaises(ValueError): rpc.request('mismatch')
        self.assertIsNotNone(rpc.process.poll())

    def test_compute_reservations_survive_unknown_failure_and_resume(self):
        self.store.reserve_compute('writer', 100, 120)
        self.store.close()
        self.store = Store(self.root/'state.sqlite',self.config,OfflineCounter())
        with self.assertRaises(BudgetExceeded): self.store.reserve_compute('writer',21,120)
        self.store.settle_compute('writer',100,60)
        self.assertEqual(self.store.stats()['compute_ledger']['writer']['charged'],60)

    def test_context_overflow_stops_before_model_call(self):
        class Tokenizer:
            def apply_chat_template(self,*args,**kwargs): return [0]*50
        client = HttpChat(ChatConfig('http://unused.invalid/v1','model',max_tokens=10))
        wrapped = TokenBudgetClient(client,Tokenizer(),self.store,'task',1000,59)
        with patch.object(client,'complete') as call:
            with self.assertRaises(ValueError): wrapped.complete([])
            call.assert_not_called()

    def test_server_token_mismatch_keeps_full_charge(self):
        class Tokenizer:
            def apply_chat_template(self,*args,**kwargs): return [0]*5
        client = HttpChat(ChatConfig('http://unused.invalid/v1','model',max_tokens=10))
        wrapped = TokenBudgetClient(client,Tokenizer(),self.store,'task',1000,100)
        def complete(*args):
            client.response_usage.value = {'prompt_tokens':6,'completion_tokens':1}
            return 'answer'
        with patch.object(client,'complete',side_effect=complete):
            with self.assertRaises(ValueError): wrapped.complete([])
        self.assertEqual(self.store.stats()['compute_ledger']['task']['charged'],15)

    def test_periodic_overload_has_finite_queue_and_counts_rejections(self):
        self.stream()
        class SlowPolicy(ToyPolicy):
            def act(self,*args,**kwargs):
                time.sleep(0.03)
                return super().act(*args,**kwargs)
        utility = utility_function({'family_scales':{'vault':1},'status_values':{'rejected':-1,'deadline':-2}})
        self.runtime = Runtime(self.store,ToyWriter(),ToyConsolidator(self.store),Validator(self.store,ToyVerifier()),utility=utility)
        report = run_arrivals(self.runtime,(self.task(i) for i in range(8)),lambda _:ToyEnvironment(),lambda _:SlowPolicy(),
                              {'concurrency':1,'queue_capacity':0,'interval_seconds':0,'task_timeout_seconds':2})
        self.assertEqual(report['offered_outcomes'],8)
        self.assertGreater(report['statuses'].get('rejected',0),0)
        self.assertEqual(self.store.stats()['completed'],8)
        self.assertEqual(self.store.stats()['stream_slots'],0)

    def test_observed_transition_verifier_rejects_generalizations_and_unrelated_actions(self):
        step = Step('You see a locked door','unlock door','The door opens',False,
                    {'native_environment':'alfworld','native_score':1,'native_score_before':0,'action_admissible':True})
        episode = Episode('a','alfworld','v1',(step,),1,True)
        claim = {'kind':'observed_transition','observation_excerpt':'locked door',
                 'action':'unlock door','feedback_excerpt':'door opens'}
        memory = Memory('m',json.dumps(claim),'Observed in alfworld; public observation contains: locked door',(Support('a',0),))
        self.assertTrue(check_transition(memory,episode,Support('a',0)))
        self.assertFalse(check_transition(replace(memory,content='Always unlock every door'),episode,Support('a',0)))
        self.assertFalse(check_transition(memory,replace(episode,steps=(replace(step,action='wait'),)),Support('a',0)))
        self.assertFalse(positive_progress(replace(episode,steps=(replace(step,evidence=dict(step.evidence,native_score=0)),)),0))
        self.assertFalse(positive_progress(replace(episode,steps=(replace(step,evidence=dict(step.evidence,native_score_before=None)),)),0))

    def test_deployment_attestation_digest_and_live_name_both_required(self):
        record = dict(model='served',revision='a'*40,tokenizer_model='tokenizer',tokenizer_revision='b'*40,context_window=1000)
        path = self.root/'deployment.json'
        path.write_text(json.dumps(record),encoding='utf-8')
        settings = dict(record,deployment_manifest=str(path),deployment_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),api_key_env='NONE')
        with patch('memturn.models.preflight.request_json',return_value={'data':[{'id':'wrong'}]}):
            with self.assertRaises(ValueError): check_deployment(settings,'http://unused.invalid/v1')
        with patch('memturn.models.preflight.request_json',return_value={'data':[{'id':'served'}]}):
            result = check_deployment(settings,'http://unused.invalid/v1')
            self.assertIn('not_weight_attestation',result['identity_evidence'])
            path.write_text('{}',encoding='utf-8')
            with self.assertRaises(ValueError): check_deployment(settings,'http://unused.invalid/v1')

    def test_adapter_recovery_is_idempotent_but_never_aliases_another_path(self):
        root = self.root/'adapters'
        root.mkdir()
        artifact = Artifact('candidate',str(root/'candidate'),'huggingface_lora','base',{})
        serving = VLLMAdapters('http://unused.invalid/v1',root,'/shared/adapters')
        with patch.object(serving,'_models',return_value={'candidate':{'root':'/shared/adapters/candidate'}}), patch.object(serving,'_post') as post:
            serving.load(artifact)
            post.assert_not_called()
        with patch.object(serving,'_models',return_value={'candidate':{'root':'/another/run'}}), patch.object(serving,'_post') as post:
            with self.assertRaises(ValueError): serving.load(artifact)
            post.assert_not_called()

    def test_remote_advertised_orphan_is_preserved_without_unloading(self):
        from memturn.learning.huggingface import HuggingFaceLearner,SFTConfig
        from unittest.mock import Mock
        serving = Mock()
        serving.is_registered.return_value = True
        learner = HuggingFaceLearner(self.root/'adapters',SFTConfig('m','revision',('q_proj',)),serving)
        orphan = learner.root/('hf_'+'a'*32)
        orphan.mkdir()
        (orphan/'weight').write_bytes(b'recoverable')
        result = learner.recover_orphans({'base'})
        self.assertEqual(result[0]['state'],'recovery_failed')
        self.assertTrue(orphan.exists())
        serving.unload.assert_not_called()

    def test_multidimensional_budget_admission_is_atomic(self):
        with self.assertRaises(BudgetExceeded):
            self.store.reserve_computes([('sft_tokens',100,1000),('sft_updates',5,4)])
        self.assertEqual(self.store.stats()['compute_ledger'],{})

    def test_matrix_keeps_shared_caps_and_exposes_explicit_ablation_changes(self):
        from memturn.matrix import build_matrix,budget_signature
        path = Path(__file__).resolve().parents[1]/'configs/scienceworld.example.json'
        profiles = build_matrix(json.loads(path.read_text(encoding='utf-8')))
        self.assertEqual(len(profiles),10)
        self.assertEqual(len({budget_signature(p) for p in profiles.values()}),1)
        self.assertEqual(profiles['baseline']['runtime']['training_material'],'mixed')
        self.assertEqual(profiles['random_selection']['runtime']['selection_policy'],'random')
        self.assertEqual(profiles['raw_experience_sft']['runtime']['min_support_tasks'],1)
        self.assertFalse(profiles['consolidation_only']['runtime']['internalization_enabled'])

    def test_invalid_budget_and_control_types_fail_before_execution(self):
        from memturn.learning.huggingface import SFTConfig
        for values in ({'memory_token_cap':float('nan')},{'max_pending_jobs':1.5},{'exit_after_training':'false'},
                       {'max_adapter_versions':4},{'seed':True}):
            with self.assertRaises(ValueError): Config(**values)
        with self.assertRaises(ValueError): SFTConfig('m','revision',('q_proj',),max_gpu_reserved_bytes=float('nan'))

    def test_rss_guard_terminates_over_budget_process(self):
        import importlib.util
        if importlib.util.find_spec('psutil') is None:
            self.skipTest('psutil is an optional runtime dependency')
        rpc = ProcessRPC([sys.executable,'-u','-c','import time; time.sleep(30)'],timeout=3,rss_limit_bytes=1)
        with self.assertRaises(MemoryError): rpc.request('memory')
        self.assertIsNotNone(rpc.process.poll())

    def test_native_reset_timeout_is_a_counted_task_outcome(self):
        from memturn.tasks.process import ProcessEnvironment
        class Writer:
            def write(self,*args): return None
        self.runtime = Runtime(self.store,Writer(),ToyConsolidator(self.store),Validator(self.store,ToyVerifier()))
        env = ProcessEnvironment({'worker_command':[sys.executable,'-u','-c','import time; time.sleep(30)'],
                                  'timeout_seconds':0.2})
        result = self.runtime.run_task(Task('timeout','vault','v1'),env,ToyPolicy())
        self.assertEqual(result.status,'deadline')
        self.assertEqual(self.store.stats()['completed'],1)
        self.assertIsNone(env.process)

    def test_raw_sft_ablation_uses_grounded_original_decisions_without_consolidation(self):
        from tests.test_core import episode
        self.store.config = replace(self.config,training_source='raw',training_material='decision_only',
                                     min_support_tasks=1,consolidation_enabled=False)
        source = episode('raw')
        self.store.record_result(source)
        raw = ToyWriter().write(source,0)
        self.store.append(raw)
        self.store.finish_source('raw')
        class Learner:
            def train(self, examples, parent_version):
                self.examples = examples
                return Artifact('trained','fixture','test',parent_version,{})
            def prepare(self,artifact): pass
            def retain(self,versions): pass
        class Forbidden:
            def consolidate(self,*args): raise AssertionError('raw SFT must not call the consolidator')
        learner = Learner()
        self.runtime = Runtime(self.store,ToyWriter(),Forbidden(),Validator(self.store,ToyVerifier()),learner)
        self.runtime.regular_maintenance()
        self.assertEqual(self.store.model_version,'trained')
        self.assertTrue(learner.examples)
        for example in learner.examples:
            self.assertEqual(example.prompt,source.steps[example.step].observation)
            self.assertEqual(example.target,source.steps[example.step].action)

    def test_no_exit_publication_keeps_the_selected_memory_and_growth_ledger(self):
        from tests.test_core import episode
        source = episode('keep')
        self.store.record_result(source)
        memory = ToyWriter().write(source,0)
        self.store.append(memory)
        self.store.publish(memory,Artifact('trained','fixture','test','base',{}),retire=False)
        self.assertEqual(self.store.model_version,'trained')
        self.assertEqual(self.store.memories()[0].ref,memory.ref)
        self.assertEqual(self.store.stats()['growth']['internalized_tokens'],0)
        self.assertFalse(self.store._get('publication')['retired'])

    def test_simple_cold_eviction_never_invokes_consolidation_or_training(self):
        from tests.test_core import episode
        self.store.config = replace(self.config,capacity_policy='evict',cold_fraction=1)
        source = episode('cold')
        self.store.record_result(source)
        memory = ToyWriter().write(source,0)
        self.store.append(memory)
        self.store.finish_source('cold')
        for i in range(4): self.store.record_result(episode(f'clock{i}'),keep_source=False)
        class Forbidden:
            def consolidate(self,*args): raise AssertionError('eviction called consolidation')
        self.runtime = Runtime(self.store,ToyWriter(),Forbidden(),Validator(self.store,ToyVerifier()))
        self.runtime.capacity_maintenance(incoming_tokens=self.config.memory_token_cap)
        self.assertFalse(self.store.memories())
        self.assertEqual(self.store.stats()['growth']['capacity_released_tokens'],self.store.counter.count(memory.payload))

    def test_timeout_kills_worker_descendants(self):
        import os
        if os.name == 'nt':
            import importlib.util
            if importlib.util.find_spec('psutil') is None: self.skipTest('psutil needed to inspect Windows child')
        worker = ('import sys,subprocess,json,time\n'
                  'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"])\n'
                  'for line in sys.stdin:\n'
                  ' r=json.loads(line)\n'
                  ' if r["op"]=="hang": time.sleep(30)\n'
                  ' print(json.dumps({"id":r["id"],"result":child.pid}),flush=True)\n')
        rpc = ProcessRPC([sys.executable,'-u','-c',worker],timeout=3)
        pid = rpc.request('spawn')
        started = time.monotonic()
        with self.assertRaises(TimeoutError): rpc.request('hang',timeout=0.2)
        self.assertLess(time.monotonic()-started,5,'cleanup waited for the child to finish naturally')
        if os.name == 'nt':
            import psutil
            self.assertFalse(psutil.pid_exists(pid))
        else:
            path = Path(f'/proc/{pid}/stat')
            self.assertTrue(not path.exists() or path.read_text().split()[2] == 'Z')


if __name__ == '__main__': unittest.main()

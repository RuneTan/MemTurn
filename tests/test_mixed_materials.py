"""Protocol, leakage, accounting and native replay checks; no task-performance claims."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

from memturn.config import Config
from memturn.contracts import Artifact, Episode, Example, ExposedMemory, Memory, RetrievalEvent, RetrievalResult, Step, Support, Task, Transition
from memturn.learning.huggingface import supervised_tokens
from memturn.learning.materials import bind_example, rule_question
from memturn.memory.store import BudgetExceeded, Store
from memturn.memory.validation import Validator
from memturn.models.agents import ChatConsolidator, ChatWriter, decode_memory
from memturn.models.http import task_messages
from memturn.runtime.service import Runtime
from memturn.tasks.process import ProcessEnvironment
from memturn.tasks.verification import NativeReplayVerifier
from memturn.tokens import OfflineCounter


class SourceVerifier:
    def verify_support(self, memory, episode, support):
        return support.task_id == episode.task_id and support.step == 0

    def verify_example(self, example, episode):
        return example.prompt == episode.steps[0].observation and example.target == episode.steps[0].action


class MixedMaterialsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.directory.name)/'state.sqlite', Config(), OfflineCounter())
        self.verifier = Validator(self.store, SourceVerifier())
        for task in ('a','b'):
            self.store.record_result(Episode(task,'scienceworld','private_dataset_label',
                (Step('Public original state','look around','Original feedback',False),),0,False,
                task_spec={'id':task,'family':'scienceworld','version':'private'}))
        memory = Memory('m','Inspect the current room before choosing a destination.','At an unfamiliar location',
                        (Support('a',0),Support('b',0)),parents=('old@1',),consolidated=True)
        decisions = tuple(replace(bind_example(Example(t,0,'Public original state','look around'),memory),
                                  support_refs=(Support(t,0),)) for t in ('a','b'))
        self.memory = replace(memory,examples=(rule_question(memory),*decisions))

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def test_mixed_requires_both_types_and_does_not_inflate_task_support(self):
        self.assertTrue(self.verifier.trainable(self.memory))
        self.assertEqual(self.verifier.support_count(self.memory),2)
        self.assertFalse(self.verifier.trainable(replace(self.memory,examples=self.memory.examples[1:])))
        self.assertFalse(self.verifier.trainable(replace(self.memory,examples=self.memory.examples[:1])))
        repeated = replace(self.memory,examples=(self.memory.examples[0],)*5+self.memory.examples[1:])
        self.assertEqual(self.verifier.support_count(repeated),2)

    def test_wrong_owner_target_source_or_answer_in_question_is_rejected(self):
        qa, decision = self.memory.examples[:2]
        for changed in (replace(qa,memory_ref='m@2'),replace(qa,target='Invented rule'),
                        replace(qa,prompt='Repeat '+qa.target),replace(qa,output_kind='environment_action'),
                        replace(qa,covered_content_refs=('other@1#content',)),
                        replace(decision,prompt='Earlier convenient state'),replace(decision,input_step=1),
                        replace(decision,support_refs=(Support('foreign',0),))):
            with self.subTest(changed=changed):
                self.assertFalse(self.verifier.valid_example(changed,self.memory))

    def test_one_call_decode_binds_qa_and_copies_native_source_verbatim(self):
        source = replace(self.store.episode('a'),steps=(Step('Native state','look around','feedback',False,
                                                          {'native_environment':'scienceworld'}),))
        value = dict(local_id='result1',content=self.memory.content,applicability=self.memory.applicability,
                     supports=[{'task_id':'a','step':0}],parents=['old@1'],examples=[
                         {'example_type':'rule_qa','question':'How should an unfamiliar location be approached?'},
                         {'example_type':'decision','task_id':'a','step':0}])
        decoded=decode_memory(value,1,'decoded',lambda _:source)
        self.assertEqual(decoded.examples[0].target,decoded.content)
        self.assertEqual(decoded.examples[1].prompt,'Native state')
        self.assertEqual(decoded.examples[1].target,'look around')
        self.assertTrue(all(e.memory_ref==decoded.ref for e in decoded.examples))
        value['examples'][0]['memory_ref']='someone_else@1'
        external_only=decode_memory(value,1,'decoded',lambda _:source)
        self.assertEqual(external_only.examples,())
        self.assertEqual(external_only.content,decoded.content)

    def test_bad_training_material_does_not_destroy_valid_external_replacement(self):
        bad = replace(self.memory,examples=self.memory.examples[1:])
        class Operator:
            def consolidate(self,*args): return (bad,)
        parent=Memory('old','Older grounded statement','A location',self.memory.supports)
        self.store.append(parent)
        runtime=Runtime(self.store,None,Operator(),self.verifier)
        try:
            proposed=runtime._propose((parent,),'regular')
            self.assertEqual(proposed[0].content,bad.content)
            self.assertEqual(proposed[0].examples,())
        finally:
            runtime.close()

    def test_writer_sees_retrieval_history_but_not_private_manifest(self):
        class Client:
            def complete(self,messages):
                self.messages=messages
                return '{"memory":null}'
        client=Client()
        episode=replace(self.store.episode('a'),retrievals=(RetrievalEvent(0,None,
            (ExposedMemory('past@1','Public recalled rule',('earlier',)),),2),))
        self.assertEqual(Episode.from_dict(episode.to_dict()),episode)
        ChatWriter(client).write(episode,0)
        public=json.loads(client.messages[-1]['content'])
        self.assertNotIn('task_spec',public)
        self.assertNotIn('condition',public)
        self.assertEqual(public['retrievals'][0]['memories'][0]['payload'],'Public recalled rule')

    def test_consolidator_source_inspection_has_no_native_dataset_label(self):
        self.store.append(self.memory)
        class Client:
            calls=0
            def complete(self,messages):
                self.calls+=1
                if self.calls==1: return '{"op":"inspect_source","task_id":"a","step":0}'
                self.reply=json.loads(messages[-1]['content'])
                return '{"op":"finish"}'
        client=Client()
        ChatConsolidator(client,self.store,self.store.counter).consolidate((self.memory,),'regular',0,2000)
        self.assertNotIn('condition',client.reply)
        self.assertNotIn('task_spec',client.reply)

    def test_rule_qa_is_text_with_masked_question_not_an_environment_command(self):
        class Tokenizer:
            def apply_chat_template(self,messages,add_generation_prompt,**kwargs):
                return ''.join(f"<{m['role']}>{m['content']}" for m in messages)+('<assistant>' if add_generation_prompt else '!')
            def encode(self,text,**kwargs): return list(text.encode())
        qa=self.memory.examples[0]
        ids,labels=supervised_tokens(Tokenizer(),qa.prompt,qa.target,4000,output_kind=qa.output_kind)
        start=next(i for i,v in enumerate(labels) if v!=-100)
        self.assertEqual(bytes(labels[start:]).decode(),qa.target+'!')
        self.assertNotIn(qa.target,bytes(ids[:start]).decode())
        self.assertNotIn('"kind": "environment"',bytes(labels[start:]).decode())

    def test_historical_retrieval_bodies_are_retained_once(self):
        a,b=self.memory,replace(self.memory,id='other',content='Second distinct recalled rule')
        history=(RetrievalResult(None,(a,),2),RetrievalResult('another query',(b,),1),
                 RetrievalResult('repeat',(a,),0))
        text=json.dumps(task_messages('Public state',(a,),history))
        self.assertEqual(text.count(a.content),1)
        self.assertEqual(text.count(b.content),1)
        self.assertIn('another query',text)
        self.assertIn('already_in_context',text)

    def test_growth_ledger_balances_admission_replacement_publish_and_rollback(self):
        self.store.append(self.memory)
        self.store.append(self.memory)  # idempotent retry
        def balance():
            stats=self.store.stats(); g=stats['growth']
            self.assertEqual(stats['active_tokens'],g['initial_tokens']+g['admitted_tokens']-
                             g['regular_released_tokens']-g['internalized_tokens']-g['capacity_released_tokens'])
        balance()
        revised=replace(self.memory,id='new',content='Inspect location.',parents=(self.memory.ref,),examples=())
        self.store.replace_region((self.memory,),(revised,),'regular'); balance()
        shorter=replace(revised,id='small',content='Look.',applicability='Here',parents=(revised.ref,))
        self.store.replace_region((revised,),(shorter,),'capacity'); balance()
        self.store.publish(shorter,Artifact('v1','unused','test','base',{})); balance()
        before=self.store.stats()['growth']
        huge=replace(self.memory,id='huge',content='enormous '*100000)
        with self.assertRaises(BudgetExceeded): self.store.append(huge)
        self.assertEqual(self.store.stats()['growth'],before); balance()


class NativeReplayTests(unittest.TestCase):
    def source(self):
        task=Task('native','scienceworld','task')
        from dataclasses import asdict
        return Episode(task.id,task.family,'not visible',(Step('room','look','room again',False,
            {'native_environment':'scienceworld','action_admissible':True}),),0,False,task_spec=asdict(task))

    def verifier(self,summary_check=None):
        class Environment:
            def reset(self,task): return 'room'
            def step(self,action): return Transition('room again',0,False,evidence={
                'native_environment':'scienceworld','action_admissible':True})
            def close(self): pass
        return NativeReplayVerifier(Environment,summary_check)

    def test_replay_does_not_self_certify_summary(self):
        ep=self.source(); verifier=self.verifier()
        self.assertTrue(verifier.replay_matches(ep))
        self.assertFalse(verifier.verify_support(None,ep,Support('native',0)))
        self.assertTrue(verifier.verify_example(Example('native',0,'room','look'),ep))

    def test_replay_rejects_tampered_observation_evidence_reward_or_success(self):
        ep=self.source(); verifier=self.verifier(lambda *args:True)
        for changed in (replace(ep,reward=1),replace(ep,success=True),
                        replace(ep,task_spec=dict(ep.task_spec,id='different_source')),
                        replace(ep,steps=(replace(ep.steps[0],observation='invented'),)),
                        replace(ep,steps=(replace(ep.steps[0],evidence={'action_admissible':True}),))):
            with self.subTest(episode=changed): self.assertFalse(verifier.replay_matches(changed))

    def test_domain_verifier_exception_rejects_support(self):
        def broken(*args): raise ValueError('unsupported domain claim')
        self.assertFalse(self.verifier(broken).verify_support(None,self.source(),Support('native',0)))

    def test_process_protocol_uses_separate_worker_and_can_restart(self):
        worker=('import sys,json; '
                '\nfor line in sys.stdin:\n'
                ' r=json.loads(line); op=r["op"]; '
                'v="room" if op=="reset" else ({"observation":"next","reward":0,"done":False} if op=="step" else None); '
                'print(json.dumps({"id":r["id"],"result":v}),flush=True); '
                '\n if op=="close": break\n')
        env=ProcessEnvironment({'worker_command':[sys.executable,'-u','-c',worker],'timeout_seconds':3})
        try:
            for _ in range(2):
                self.assertEqual(env.reset(Task('x','scienceworld','v')),'room')
                self.assertEqual(env.step('look').observation,'next')
                env.close()
        finally: env.close(force=True)

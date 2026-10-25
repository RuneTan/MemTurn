"""Local HTTP/native-process fixtures exercise the real remote entry and resume."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from memturn.experiments import run_profile
from memturn.tokens import OfflineCounter


class RemoteAssemblyTests(unittest.TestCase):
    def test_real_http_subprocesses_stream_commit_and_resume_without_reexecution(self):
        counts = {'task':0,'writer':0}
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_GET(self):
                self.reply({'data':[{'id':'fixture-'+role} for role in ('task','writer','consolidator')]})
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                role = request['model'].removeprefix('fixture-')
                if role in counts: counts[role] += 1
                text = '{"memory":null}' if role == 'writer' else '{"kind":"environment","action":"look"}'
                self.reply({'choices':[{'message':{'content':text}}], 'usage':{'prompt_tokens':10,'completion_tokens':5}})
            def reply(self,value):
                body = json.dumps(value).encode()
                self.send_response(200)
                self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                manifest = root/'tasks.jsonl'
                manifest.write_text(''.join(json.dumps({'id':str(i),'family':'scienceworld','version':'fixture',
                    'max_steps':2})+'\n' for i in range(2)),encoding='utf-8')
                worker = (
                    'import sys,json\n'
                    'for line in sys.stdin:\n'
                    ' r=json.loads(line)\n'
                    ' result="public fixture observation" if r["op"]=="reset" else '
                    '{"observation":"finished", "reward":1.0,"done":True,"success":True} if r["op"]=="step" else {}\n'
                    ' print(json.dumps({"id":r["id"],"result":result}),flush=True)\n')
                profile = {'operator_kind':'fixture_operators_not_native_learning','runtime':{},'stream_id':'fixture',
                    'arrival_policy':'wait_for_background','arrival':{'task_timeout_seconds':10},'tokenizer_model':'fixture','tokenizer_revision':'a'*40,
                    'verification_factory':'memturn.tasks.verification:BenchmarkVerifierRequired',
                    'verification_kwargs':{},'models':{},
                    'environment':{'kind':'scienceworld','task_manifest':str(manifest),
                        'worker_command':[sys.executable,'-u','-c',worker], 'timeout_seconds':5},
                    'training':{'enabled':False},'resources':{'training_rss_limit_bytes':1073741824,'active_state_disk_bytes':268435456},
                    'learning_budget':{'tokens':1000,'updates':10},
                    'utility':{'family_scales':{'scienceworld':1},'status_values':{'deadline':-1,'rejected':-1,'execution_error':-1}}}
                profile['verification_factory'] = 'memturn.tasks.observed:make_verifier'
                profile['verification_kwargs'] = {'environment':profile['environment']}
                for role in ('task','writer','consolidator'):
                    record = dict(model='fixture-'+role,revision='a'*40,tokenizer_model='fixture',tokenizer_revision='a'*40,context_window=1000)
                    path = root/(role+'.json')
                    path.write_text(json.dumps(record),encoding='utf-8')
                    profile['models'][role] = dict(record,base_url_env='MEMTURN_TEST_ENDPOINT',api_key_env='MEMTURN_TEST_UNUSED_KEY',
                        max_tokens=20,total_token_budget=10000,deployment_manifest=str(path),
                        deployment_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                profile_path = root/'profile.json'
                profile_path.write_text(json.dumps(profile),encoding='utf-8')
                tokenizer = SimpleNamespace(apply_chat_template=lambda *args,**kwargs:[0]*10)
                transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args,**kwargs:tokenizer))
                sampler = SimpleNamespace(report=lambda:{'fixture':True},close=lambda:None)
                with patch.dict(os.environ,{'MEMTURN_TEST_ENDPOINT':f'http://127.0.0.1:{server.server_port}/v1'}), \
                     patch.dict(sys.modules,{'transformers':transformers}), \
                     patch('memturn.experiments.ResourceSampler',return_value=sampler), \
                     patch('memturn.experiments.HuggingFaceCounter',return_value=OfflineCounter()):
                    protected = root/'unrelated'
                    protected.mkdir()
                    (protected/'keep.txt').write_text('preserve',encoding='utf-8')
                    with self.assertRaisesRegex(ValueError,'must be empty'):
                        run_profile(profile_path,protected)
                    self.assertEqual((protected/'keep.txt').read_text(encoding='utf-8'),'preserve')
                    first = run_profile(profile_path,root/'run')
                    self.assertEqual(first['stats']['completed'],2)
                    diagnostic = [json.loads(line) for line in (root/'run/audit.jsonl').read_text(encoding='utf-8').splitlines()
                                  if json.loads(line).get('event') in ('task_error','task_completed','background_error')]
                    self.assertEqual(first['stats']['utility'],2, diagnostic)
                    self.assertEqual(first['errors'],[])
                    calls = dict(counts)
                    second = run_profile(profile_path,root/'run',resume=True)
                    self.assertEqual(second['stats']['completed'],2)
                    self.assertEqual(second['stats']['stream']['high_watermark'],1)
                    self.assertEqual(counts['writer'],calls['writer'])
                    self.assertEqual(counts['task'],calls['task']+1,'resume runs one protocol probe and no old task')
                    self.assertEqual(second['stats']['compute_ledger']['task']['confirmed'],60)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)


if __name__ == '__main__': unittest.main()

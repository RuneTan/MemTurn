"""Remote assembly with explicit identity, bounded recovery and service accounting."""
from collections import Counter
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

from .config import Config
from .contracts import dumps
from .learning.huggingface import SFTConfig
from .learning.isolated import IsolatedHuggingFaceLearner
from .memory.store import Store
from .memory.validation import Validator
from .models.agents import ChatConsolidator, ChatWriter
from .models.budget import TokenBudgetClient
from .models.http import ChatConfig, ChatPolicy, HttpChat
from .models.preflight import check_deployment, protocol_probe
from .models.publication import VLLMAdapters
from .runtime.arrival import run_arrivals, utility_function
from .runtime.recovery import RunLease, inspect_manifest, read_manifest
from .runtime.resources import hardware_report, check_resource_reservation, ResourceSampler
from .runtime.service import Runtime
from .tasks.isolated import ProcessVerifier
from .tokens import HuggingFaceCounter


def inspect_profile(profile):
    missing = []
    def visit(value, path=''):
        if isinstance(value, dict):
            for key, child in value.items(): visit(child, f'{path}.{key}'.lstrip('.'))
        elif isinstance(value, list):
            for index, child in enumerate(value): visit(child, f'{path}[{index}]')
        elif isinstance(value, str) and value.startswith('TO_'):
            missing.append(path)
    visit(profile)
    Config(**profile['runtime'])
    for key in ('stream_id','arrival','utility','resources','learning_budget','verification_factory'):
        if key not in profile: missing.append(key)
    if profile.get('training', {}).get('enabled') and not profile['training'].get('target_modules'):
        missing.append('training.target_modules')
    if not profile.get('environment', {}).get('worker_command'):
        missing.append('environment.worker_command')
    if not profile.get('arrival', {}).get('task_timeout_seconds'):
        missing.append('arrival.task_timeout_seconds')
    for role, settings in profile['models'].items():
        for key in ('revision','tokenizer_model','tokenizer_revision','context_window','total_token_budget',
                    'deployment_manifest','deployment_manifest_sha256'):
            if key not in settings: missing.append(f'models.{role}.{key}')
        if not os.environ.get(settings['base_url_env']):
            missing.append(f"models.{role}.base_url_env:{settings['base_url_env']}")
    if profile.get('verification_factory', '').endswith('BenchmarkVerifierRequired'):
        missing.append('verification_factory')
    if not missing:
        validate_profile_values(profile)
    return sorted(set(missing))


def validate_profile_values(profile):
    if set(profile['models']) != {'task','writer','consolidator'}:
        raise ValueError('the profile requires exactly the task, writer and consolidator roles')
    for role, settings in profile['models'].items():
        for name in ('context_window','total_token_budget'):
            value = settings[name]
            if isinstance(value,bool) or not isinstance(value,int) or value <= 0:
                raise ValueError(f'models.{role}.{name} must be a positive integer')
        for name in ('revision','tokenizer_revision'):
            value = settings[name]
            if len(value) != 40 or any(char not in '0123456789abcdef' for char in value):
                raise ValueError('model and tokenizer revisions must be immutable 40-character commit hashes')
    task = profile['models']['task']
    if (profile['tokenizer_model'],profile['tokenizer_revision']) != (task['tokenizer_model'],task['tokenizer_revision']):
        raise ValueError('memory budgets must use the frozen task-model tokenizer')
    for group in ('family_scales','status_values'):
        if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in profile['utility'][group].values()):
            raise ValueError('utility rules must contain finite numeric values')
    for value in (*profile['resources'].values(), *profile['learning_budget'].values()):
        if isinstance(value,bool) or not isinstance(value,int) or value <= 0:
            raise ValueError('resource and learning budgets must be positive integers')
    timeout = profile['arrival']['task_timeout_seconds']
    if not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('task timeout must be finite and positive')
    if profile['training']['enabled']:
        settings = profile['training']
        if settings['model_revision'] != task['revision']:
            raise ValueError('training and serving must use the same pinned base revision')
        if settings.get('steps') is not None or settings.get('gradient_accumulation',1) != 1:
            raise ValueError('remote profiles require complete passes with gradient_accumulation=1')


def environment_for(settings):
    if settings.get('worker_command'):
        from .tasks.process import ProcessEnvironment
        return ProcessEnvironment(settings)
    if settings.get('package_version'):
        import importlib.metadata
        if importlib.metadata.version(settings['kind']) != settings['package_version']:
            raise ValueError('native environment package differs from the pinned version')
    if settings['kind'] == 'scienceworld':
        from .tasks.scienceworld import ScienceWorldEnvironment
        return ScienceWorldEnvironment(jar_path=settings.get('jar_path'),
                                       simplifications=settings.get('simplifications', ''))
    if settings['kind'] == 'alfworld':
        from .tasks.alfworld import ALFWorldEnvironment
        return ALFWorldEnvironment(settings['config_path'], settings['split'])
    raise ValueError('Only ScienceWorld and ALFWorld are in the first benchmark phase')


def run_profile(profile_path, output, resume=False):
    profile = json.loads(Path(profile_path).read_text(encoding='utf-8'))
    missing = inspect_profile(profile)
    if missing:
        raise ValueError('Complete remote setup first: ' + ', '.join(missing))
    config = Config(**profile['runtime'])
    manifest = inspect_manifest(profile['environment']['task_manifest'], expected_family=profile['environment']['kind'])
    digest = hashlib.sha256(dumps(profile).encode()).hexdigest()
    root = Path(output).resolve()
    if not resume and root.exists() and any(path.name != 'run.lock' for path in root.iterdir()):
        raise ValueError('new run directory must be empty; preserve existing files or use explicit --resume')
    root.mkdir(parents=True, exist_ok=True)
    with RunLease(root / 'run.lock'):
        exists = (root / 'state.sqlite').exists()
        if exists != resume:
            raise ValueError('existing runs require --resume; new runs must omit it')
        if resume:
            old = json.loads((root / 'profile.json').read_text(encoding='utf-8'))
            if hashlib.sha256(dumps(old).encode()).hexdigest() != digest:
                raise ValueError('resume requires an identical frozen profile')
        reservation = check_resource_reservation(profile, config, root)
        deployment = {role:check_deployment(settings, os.environ[settings['base_url_env']])
                      for role, settings in profile['models'].items()}
        (root / 'profile.json').write_text(dumps(profile), encoding='utf-8')
        hardware = hardware_report(root)
        counter = HuggingFaceCounter(profile['tokenizer_model'], profile['tokenizer_revision'])
        store = Store(root / 'state.sqlite', config, counter)
        runtime, sampler = None, None
        counts = Counter()
        try:
            sampler = ResourceSampler()
            store.configure_stream(profile['stream_id'], manifest['sha256'])
            store.recover_stream()
            from transformers import AutoTokenizer
            clients = {}
            for role, settings in profile['models'].items():
                values = {key:value for key,value in settings.items() if key in {f.name for f in fields(ChatConfig)}}
                values.update(base_url=os.environ[settings['base_url_env']], isolate_process=True)
                client = HttpChat(ChatConfig(**values))
                tokenizer = AutoTokenizer.from_pretrained(settings['tokenizer_model'], revision=settings['tokenizer_revision'])
                clients[role] = TokenBudgetClient(client, tokenizer, store, role,
                    settings['total_token_budget'], settings['context_window'])
            verifier = ProcessVerifier(profile['verification_factory'], profile.get('verification_kwargs'),
                timeout=profile.get('verification_timeout_seconds', 120), max_bytes=config.source_training_byte_cap)
            learner = None
            if profile['training']['enabled'] and config.internalization_enabled:
                settings = profile['training']
                serving = VLLMAdapters(os.environ[profile['models']['task']['base_url_env']], root / 'adapters',
                    settings['server_adapter_root'], profile['models']['task']['api_key_env'],
                    probe=lambda version:protocol_probe(clients['task'], version))
                values = {key:value for key,value in settings.items() if key in {f.name for f in fields(SFTConfig)}}
                values['target_modules'] = tuple(values['target_modules'])
                learner = IsolatedHuggingFaceLearner(root / 'adapters', SFTConfig(**values), serving,
                    rss_limit_bytes=profile['resources']['training_rss_limit_bytes'],
                    job_byte_cap=config.source_training_byte_cap)
            with (root / 'audit.jsonl').open('a' if resume else 'w', encoding='utf-8') as stream:
                audit_lock = threading.Lock()
                def audit(event):
                    with audit_lock:
                        counts[event['event']] += 1
                        stream.write(dumps(event) + '\n')
                        stream.flush()
                audit({'event':'service_segment_started', 'resume':resume, 'manifest':manifest,
                       'profile_sha256':digest, 'hardware':hardware, 'deployment':deployment, 'reservation':reservation})
                protocol_probe(clients['task'])
                guidance = profile.get('operator_guidance', '')
                if profile['verification_factory'] == 'memturn.tasks.observed:make_verifier':
                    from .tasks.observed import GUIDANCE
                    guidance += ' ' + GUIDANCE
                runtime = Runtime(store, ChatWriter(clients['writer'], guidance),
                    ChatConsolidator(clients['consolidator'], store, counter, task_guidance=guidance),
                    Validator(store, verifier), learner, audit, utility=utility_function(profile['utility']),
                    learning_budget=profile['learning_budget'])
                try:
                    if profile['arrival_policy'] == 'periodic':
                        service = run_arrivals(runtime, read_manifest(profile['environment']['task_manifest']),
                            lambda task:environment_for(profile['environment']), lambda task:ChatPolicy(clients['task']), profile['arrival'])
                    elif profile['arrival_policy'] == 'wait_for_background':
                        for sequence, task in enumerate(read_manifest(profile['environment']['task_manifest'])):
                            runtime.run_task(task, environment_for(profile['environment']), ChatPolicy(clients['task']), sequence,
                                             deadline=time.monotonic()+profile['arrival']['task_timeout_seconds'])
                            runtime.drain()
                        service = {'scope':'sequential_development_only_not_service_SLA'}
                    else:
                        raise ValueError('unknown arrival policy')
                finally:
                    runtime.close()
                report = {'stats':store.stats(), 'events':dict(counts), 'errors':runtime.errors,
                          'service':service, 'model_usage_segment':{role:client.usage for role,client in clients.items()},
                          'operator_kind':profile['operator_kind'], 'arrival_policy':profile['arrival_policy'],
                          'hardware':hardware, 'deployment':deployment, 'reservation':reservation,
                          'resource_samples':sampler.report(),
                          'profile_sha256':digest, 'manifest':manifest}
                (root / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
                return report
        finally:
            if runtime is not None: runtime.close()
            if sampler is not None: sampler.close()
            store.close()

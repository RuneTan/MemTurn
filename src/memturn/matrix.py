"""Explicit ablation profiles; generating them does not run an experiment."""
from copy import deepcopy
import hashlib

from .config import Config
from .contracts import dumps


ARMS = {
    'baseline':{},
    'random_selection':{'selection_policy':'random'},
    'consolidation_only':{'internalization_enabled':False},
    'train_without_exit':{'exit_after_training':False},
    'qa_only':{'training_material':'qa_only'},
    'decision_only':{'training_material':'decision_only'},
    'cold_eviction':{'capacity_policy':'evict'},
    'raw_experience_sft':{'consolidation_enabled':False, 'training_source':'raw',
                          'training_material':'decision_only', 'min_support_tasks':1},
    'raw_external_only':{'consolidation_enabled':False,'internalization_enabled':False,'capacity_policy':'evict'},
    'no_memory':{'memory_enabled':False,'writer_enabled':False,'internalization_enabled':False},
}


def budget_signature(profile):
    runtime = Config(**profile['runtime']).to_dict()
    for key in set().union(*(set(values) for values in ARMS.values())):
        runtime.pop(key, None)
    fixed = {key:profile.get(key) for key in ('models','arrival','utility','resources','learning_budget','verification_factory',
                                           'verification_kwargs','operator_kind','operator_guidance',
                                           'environment','tokenizer_model','tokenizer_revision')}
    fixed['runtime'] = runtime
    fixed['training'] = {key:value for key,value in profile['training'].items() if key != 'server_adapter_root'}
    return hashlib.sha256(dumps(fixed).encode()).hexdigest()


def build_matrix(profile):
    profiles = {}
    for name, overrides in ARMS.items():
        candidate = deepcopy(profile)
        candidate['runtime'] = dict(Config(**profile['runtime']).to_dict(), **overrides)
        Config(**candidate['runtime'])
        candidate['experiment_arm'] = name
        candidate['training']['server_adapter_root'] = 'TO_SET_SHARED_ADAPTER_DIRECTORY_FOR_' + name.upper()
        profiles[name] = candidate
    if len({budget_signature(p) for p in profiles.values()}) != 1:
        raise ValueError('ablation generation changed a shared budget or workload')
    return profiles

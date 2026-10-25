"""Keep GPU training outside the service process and kill overdue candidates."""
from dataclasses import asdict
import shutil

from ..contracts import Artifact
from ..contracts import dumps
from ..runtime.process import run_worker
from .huggingface import HuggingFaceLearner


class IsolatedHuggingFaceLearner(HuggingFaceLearner):
    def __init__(self, directory, config, serving, rss_limit_bytes=None, job_byte_cap=33554432):
        super().__init__(directory, config, serving)
        self.rss_limit_bytes, self.job_byte_cap = rss_limit_bytes, job_byte_cap

    def train(self, examples, parent_version, memory_context=(), histories=None):
        if not self.has_artifact_capacity(3):
            raise ValueError('adapter directory limit reached')
        if shutil.disk_usage(self.root).free < self.config.artifact_byte_cap:
            raise ValueError('insufficient disk space for the full candidate reservation')
        parent = None if parent_version == 'base' else self.artifacts[parent_version]
        response = run_worker('train', {'root':str(self.root), 'config':asdict(self.config),
            'parent':asdict(parent) if parent else None, 'parent_version':parent_version,
            'examples':[asdict(e) for e in examples], 'memories':[m.to_dict() for m in memory_context],
            'histories':[{'task_id':key[0], 'step':key[1], 'history':[asdict(s) for s in history]}
                         for key, history in (histories or {}).items()]},
            timeout=self.config.max_seconds, max_bytes=self.job_byte_cap,
            rss_limit_bytes=self.rss_limit_bytes, return_metrics=True)
        artifact = Artifact(**response['result'])
        self._validate_artifact(artifact)
        artifact.metrics['worker_peak_process_tree_rss_bytes'] = response['peak_process_tree_rss_bytes']
        artifact.metrics['worker_wall_seconds'] = response['wall_seconds']
        (self.root/artifact.version/'manifest.json').write_text(dumps(asdict(artifact)),encoding='utf-8')
        self.artifacts[artifact.version] = artifact
        return artifact

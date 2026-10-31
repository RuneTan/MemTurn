"""Explicit budgets shared by storage, retrieval, selection and maintenance."""
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    memory_token_cap: int = 12000
    source_training_byte_cap: int = 33554432
    metadata_byte_cap: int = 8388608
    snapshot_byte_cap: int = 67108864
    database_byte_cap: int = 67108864
    database_total_byte_cap: int = 201326592
    writer_timeout_seconds: float = 120.0
    consolidation_timeout_seconds: float = 300.0
    batch_size: int = 4
    region_token_cap: int = 4096
    output_token_cap: int = 2048
    entry_token_cap: int = 512
    min_support_tasks: int = 2
    training_material: str = "mixed"
    selection_policy: str = "support"
    capacity_policy: str = "consolidate"
    training_source: str = "consolidated"
    consolidation_enabled: bool = True
    internalization_enabled: bool = True
    exit_after_training: bool = True
    memory_enabled: bool = True
    writer_enabled: bool = True
    max_examples_per_job: int = 32
    retrieval_top_k: int = 3
    retrieval_token_cap: int = 768
    max_extra_searches: int = 2
    capacity_interval_batches: int = 4
    cold_fraction: float = 0.2
    return_token_ratio: float = 0.5
    max_capacity_regions: int = 2
    max_pending_jobs: int = 2
    max_sessions: int = 4
    max_adapter_versions: int = 3
    max_job_records: int = 64
    stream_window: int = 32
    seed: int = 42

    def __post_init__(self):
        for name in ("memory_token_cap", "source_training_byte_cap", "metadata_byte_cap",
                     "snapshot_byte_cap", "database_byte_cap", "database_total_byte_cap",
                     "writer_timeout_seconds", "consolidation_timeout_seconds", "batch_size", "region_token_cap",
                     "output_token_cap", "entry_token_cap", "min_support_tasks", "retrieval_top_k",
                     "max_examples_per_job",
                     "retrieval_token_cap", "max_capacity_regions", "max_pending_jobs", "max_sessions",
                     "max_adapter_versions", "max_job_records", "stream_window"):
            if (not isinstance(getattr(self,name),(int,float)) or isinstance(getattr(self,name),bool)
                    or not math.isfinite(getattr(self,name)) or getattr(self, name) <= 0):
                raise ValueError(f"{name} must be positive")
            if name not in ('writer_timeout_seconds','consolidation_timeout_seconds') and not isinstance(getattr(self,name),int):
                raise ValueError(f'{name} must be an integer')
        if (isinstance(self.capacity_interval_batches, bool)
                or not isinstance(self.capacity_interval_batches, int)
                or self.capacity_interval_batches <= 1):
            raise ValueError("capacity_interval_batches must be an integer greater than 1")
        if not 0 < self.cold_fraction <= 1 or self.return_token_ratio != 0.5:
            raise ValueError("v0.2 requires a positive cold_fraction and a fixed half return budget")
        if (isinstance(self.max_extra_searches,bool) or not isinstance(self.max_extra_searches,int)
                or self.max_extra_searches < 0 or self.max_adapter_versions != 3):
            raise ValueError("Invalid search or adapter-version limit")
        if self.training_material not in ('mixed', 'decision_only', 'qa_only'):
            raise ValueError('training_material must be mixed, decision_only or qa_only')
        if self.selection_policy not in ('support', 'random') or self.capacity_policy not in ('consolidate', 'evict'):
            raise ValueError('unknown explicitly configured ablation policy')
        if self.training_source not in ('consolidated', 'raw'):
            raise ValueError('unknown training source')
        if self.training_source == 'raw' and (self.training_material != 'decision_only' or self.min_support_tasks != 1):
            raise ValueError('raw experience ablation requires decision_only and min_support_tasks=1')
        if self.database_total_byte_cap < 3 * self.database_byte_cap:
            raise ValueError('reserve three main-database caps for main/WAL/SHM transaction peaks')
        if not isinstance(self.seed,int) or isinstance(self.seed,bool):
            raise ValueError('seed must be an integer')
        for name in ('consolidation_enabled','internalization_enabled','exit_after_training','memory_enabled','writer_enabled'):
            if not isinstance(getattr(self,name),bool):
                raise ValueError(f'{name} must be a boolean')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self):
        return asdict(self)

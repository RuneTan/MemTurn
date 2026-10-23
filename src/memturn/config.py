"""Explicit budgets shared by storage, retrieval, selection and maintenance."""
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


@dataclass(frozen=True)
class Config:
    memory_token_cap: int = 12000
    source_training_byte_cap: int = 33554432
    metadata_byte_cap: int = 8388608
    snapshot_byte_cap: int = 67108864
    database_byte_cap: int = 67108864
    batch_size: int = 4
    region_token_cap: int = 4096
    output_token_cap: int = 2048
    entry_token_cap: int = 512
    min_support_tasks: int = 2
    retrieval_top_k: int = 3
    retrieval_token_cap: int = 768
    max_extra_searches: int = 2
    capacity_trigger_ratio: float = 0.9
    capacity_target_ratio: float = 0.8
    cold_fraction: float = 0.2
    return_token_ratio: float = 0.5
    max_capacity_regions: int = 2
    max_pending_jobs: int = 2
    max_sessions: int = 4
    max_adapter_versions: int = 3
    max_job_records: int = 64
    seed: int = 42

    def __post_init__(self):
        for name in ("memory_token_cap", "source_training_byte_cap", "metadata_byte_cap",
                     "snapshot_byte_cap", "database_byte_cap", "batch_size", "region_token_cap",
                     "output_token_cap", "entry_token_cap", "min_support_tasks", "retrieval_top_k",
                     "retrieval_token_cap", "max_capacity_regions", "max_pending_jobs", "max_sessions",
                     "max_adapter_versions", "max_job_records"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.capacity_target_ratio < self.capacity_trigger_ratio <= 1:
            raise ValueError("Require 0 < capacity_target_ratio < capacity_trigger_ratio <= 1")
        if not 0 < self.cold_fraction <= 1 or self.return_token_ratio != 0.5:
            raise ValueError("v0.2 requires a positive cold_fraction and a fixed half return budget")
        if self.max_extra_searches < 0 or self.max_adapter_versions < 3:
            raise ValueError("Invalid search or adapter-version limit")

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

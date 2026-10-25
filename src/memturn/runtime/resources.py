"""Portable observations; cluster hard limits require deployment evidence."""
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess
import threading


class ResourceSampler:
    """Bounded sampled RSS aggregate, not a kernel-level memory quota."""
    def __init__(self, interval=0.25):
        import psutil
        self.psutil, self.process = psutil, psutil.Process()
        self.interval, self.stop = interval, threading.Event()
        self.peak, self.samples, self.failures = 0, 0, 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            try:
                total = 0
                for process in (self.process, *self.process.children(recursive=True)):
                    try: total += process.memory_info().rss
                    except self.psutil.NoSuchProcess: pass
                self.peak = max(self.peak,total)
                self.samples += 1
            except Exception:
                self.failures += 1
            self.stop.wait(self.interval)

    def report(self):
        return {'sampled_peak_process_tree_rss_bytes':self.peak,'sampling_interval_seconds':self.interval,
                'samples':self.samples,'sampling_failures':self.failures,
                'scope':'sum_of_RSS_including_shared_pages; sampling_can_miss_short_peaks'}

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)


def hardware_report(path='.'):
    result = {'python':platform.python_version(), 'platform':platform.platform(), 'packages':{},
              'disk':dict(zip(('total','used','free'), shutil.disk_usage(path))),
              'scheduler':{name:os.environ[name] for name in ('SLURM_JOB_ID','SLURM_CPUS_PER_TASK',
                  'SLURM_MEM_PER_NODE','CUDA_VISIBLE_DEVICES') if name in os.environ},
              'cgroup_memory_max':None, 'gpus':None}
    for package in ('torch','transformers','peft','accelerate','psutil','vllm','scienceworld','alfworld'):
        try: result['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: result['packages'][package] = None
    if Path('/proc/self/cgroup').exists():
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            if line.startswith('0::'):
                candidate = Path('/sys/fs/cgroup') / line[3:].lstrip('/') / 'memory.max'
                if candidate.is_file(): result['cgroup_memory_max'] = candidate.read_text().strip()
    if shutil.which('nvidia-smi'):
        try:
            completed = subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,driver_version',
                                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
            result['gpus'] = completed.stdout.strip().splitlines() if completed.returncode == 0 else None
        except subprocess.TimeoutExpired:
            result['gpu_query_status'] = 'timeout'
    return result


def check_resource_reservation(profile, config, root):
    resources, training = profile['resources'], profile['training']
    adapters = config.max_adapter_versions * training.get('artifact_byte_cap', 67108864) if training['enabled'] else 0
    disk = config.database_total_byte_cap + adapters
    if resources['active_state_disk_bytes'] < disk:
        raise ValueError('active-state disk limit cannot cover database and adapter reservations')
    if shutil.disk_usage(root).free < disk:
        raise ValueError('free disk cannot cover the active-state reservation')
    return {'database_reserved_bytes':config.database_total_byte_cap, 'adapter_reserved_bytes':adapters,
            'active_state_reserved_bytes':disk,
            'training_rss_limit_bytes':resources['training_rss_limit_bytes'],
            'enforcement':'process_RSS_monitor_and_CUDA_allocator_limit; cluster_quotas_require_remote_acceptance'}

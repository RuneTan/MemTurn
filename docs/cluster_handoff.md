# MemTurn cluster setup and engineering handoff

[English](cluster_handoff.md) | [简体中文](cluster_handoff.zh-CN.md)

Date: 2026-10-24. This guide supports deployment review, installation and acceptance before formal models and experiments are selected. **It documents an engineering delivery, not established benchmark gains or cluster service performance.**

## Research baseline

The default remains one user, one task stream, one bounded external bank, a frozen base model and fixed-rank LoRA. Writer records only the current episode. The same Consolidator produces memory summaries and associated training material. A regular batch selects at most one eligible consolidated entry, prioritizing distinct task support. Default mixed material contains one rule question and real source decisions. The student excludes the taught memory and identified same-source paths, traverses all eligible samples, and retires only the selected version after publication. Capacity maintenance uses a cold region, half of its actual token budget and a variable return count `k`, without training on that event.

No independent teacher, reinforcement learning, growth-adaptive scheduler or dedicated anti-forgetting module was added. Random selection, QA-only training and retention after training are explicit experimental switches, not changes to the default baseline.

## Integrated engineering work

| Area | Implemented behavior | Validation boundary |
| --- | --- | --- |
| Task-stream recovery | Frozen manifest hash, contiguous completion watermark and bounded out-of-order window; results, utility, pending sources and completion state commit together | Completed positions are not counted again after recent results rotate; interrupted simulator tasks retry from reset, without guaranteeing exactly-once external-world actions |
| Exclusive run ownership | OS file lock released on process exit; one remote coordinator per run | Lock age or a stale-looking PID does not authorize deletion |
| Process deadlines | Bounded JSON workers for HTTP, native environments, verification and SFT; timeout terminates the process tree | Linux uses process groups; Windows/nested workers also use psutil. A remote server still schedules computation it has received |
| Bounded arrivals | Fixed arrival interval, concurrency and queue limits; queue time counts toward deadlines; rejection/timeout receive results and utility | Resume starts a new service-measurement segment. Downtime is not counted as service, and p95 is a histogram bucket upper bound |
| Inference budgets | Reserve exact pinned-chat-template input plus maximum output; reject inconsistent server token accounting; persist quotas | Unknown failures/crashes retain the full reservation. Conservative charges and confirmed use are separate; cached tokens are not free computation |
| Training budgets | Persist token/update quotas, reserve jointly and settle actual completed use | A complete sample pass remains necessary for publication; exhausted quota does not silently truncate samples or passes |
| Memory and disk | Payload/tensor reservations, training-process-tree RSS sampling, CUDA allocator limits/peaks, main/WAL/SHM reservations and artifact limits | RSS sampling can double-count shared pages and miss short peaks; cluster quotas must enforce hard process and serving GPU limits |
| Models and protocols | Freeze deployment-manifest SHA-256; check served names, immutable LoRA names and shared paths; probe the action protocol after loading | The manifest is operator attestation, not cryptographic proof of remote weights; protocol success is not mastery |
| Native evidence | Conservative `observed_transition` checks and full replay; custom predicates can run in isolated processes | Certifies observed transitions with native score progress, not unrestricted generalizations or optimal actions |
| Comparison tools | Generate, execute and summarize explicit arms; check shared budgets and workload | Equal caps do not imply equal actual training. Summaries expose use but do not establish significance or mechanisms |
| Orphan recovery | Archive and verify before cleanup; protect current/in-flight versions; allow manual restore | Unknown candidates still advertised remotely are checked read-only and preserved for the deployment owner, never automatically unloaded |

## First checks without a model

Use an isolated Python 3.10+ environment from the `MemTurn` directory:

```bash
python -X utf8 scripts/cluster_inventory.py --output cluster_inventory.json
python -X utf8 run_memturn.py doctor
python -X utf8 -m unittest discover -s tests -t . -v
python -X utf8 run_memturn.py demo --config configs/smoke.json --output runs/first_offline_check
```

The inventory script reads hardware, software, disk, configured scheduler resources and GPU information. It neither prints keys nor downloads models. Test HTTP servers listen only on loopback. Missing PyTorch/training dependencies/psutil cause the corresponding optional tests to skip; retain the full log and do not count skips as passes.

Remote execution needs the project's `runtime` extra, including psutil. Install training dependencies after validating the cluster CUDA/driver combination:

```bash
python -m pip install -e '.[runtime]'
# After installing training dependencies for the validated CUDA/PyTorch combination:
python -m pip install -e '.[train]'
python -m pip freeze > cluster-resolved-requirements.txt
```

`configs/local-windows.lock.txt` records a historical laptop environment, not a Linux cluster lockfile. Native environments may use a separate Python installation; configure its interpreter and `scripts/native_worker.py` in `environment.worker_command`. The worker checks package versions, and the task manifest fixes instances, versions, seeds and step limits. Environment assets and model weights are not bundled.

## Configuration and execution

`configs/scienceworld.example.json` and `configs/alfworld.example.json` retain `TO_*` placeholders. Formal models, devices, quotas, load and utility values are undecided; numeric development defaults are not optimized settings. Copy a template and supply:

1. Served names, immutable model/tokenizer revisions, context limits and total token caps for all three roles.
2. Deployment manifests and SHA-256 values for each service, following `configs/deployment.example.json`; record startup arguments without keys and the server version.
3. The task-model loader, supported LoRA target modules and shared adapter path. Training preserves the base revision, rank, alpha and targets.
4. Native task manifest, package versions, process deadlines and a validated domain verifier.
5. Arrival interval, task deadline, concurrency, queue capacity, per-family utility normalization and failure/rejection/timeout values.
6. RAM/GPU/storage quotas and total training-token/update caps. Each run uses its own directory.

Roles may use separate endpoints. Keys enter only through the environment variable named by `api_key_env`. Training and serving must access the configured shared adapter path.

```bash
python -X utf8 run_memturn.py check-profile --profile configs/verified_profile.json
python -X utf8 run_memturn.py run --profile configs/verified_profile.json --output runs/experiment_a
python -X utf8 run_memturn.py run --profile configs/verified_profile.json --output runs/experiment_a --resume
```

Missing settings produce exit code 2. A complete static profile does not establish remote acceptance. Resume requires the same configuration and manifest; it never guesses the position of a new stream from an old task count. Use reliable filesystem locks and SQLite support, and never let multiple machines write one state database. Scheduler placeholders in `scripts/slurm_run.template.sh` require cluster-specific values.

Only `arrival_policy=periodic` uses the arrival/queue driver. `wait_for_background` is a sequential development mode and cannot support uninterrupted-service claims. Queue plus concurrency must be smaller than the recovery window. An unfinished early task limits control-record growth; later arrivals already past their deadlines still receive timeout outcomes.

## Native verifier boundaries

The built-in narrow factory is `memturn.tasks.observed:make_verifier`; its `verification_kwargs` must include `environment`. Memory content is an explicitly marked `observed_transition` record containing source observation, action and feedback excerpts, with a prescribed applicability format. Writer/Consolidator automatically receive the corresponding format guidance.

Acceptance requires consistent sources, an admissible action, positive native score change at that step and matching full replay. Multiple sources must support the same transition. This proves that the transition occurred in those sources, not that every future state matching an excerpt should use that action. Native observations are not rewritten and answer hints are not added to help a small model.

Broader rules need domain predicates through the same process-isolation and source-checking interface. Without reliable checks, writing/training is rejected. Whether to start with this narrow subset or extend it remains an experimental choice. Formal native-environment effectiveness experiments have not run.

## Comparison configurations and actual budgets

```bash
python -X utf8 scripts/build_experiment_matrix.py --profile configs/verified_profile.json --output configs/review_matrix
# Complete each arm's shared adapter path and other required fields, then select arms explicitly:
python -X utf8 scripts/run_experiment_matrix.py --matrix configs/review_matrix --output runs/matrix --arms baseline random_selection
python -X utf8 scripts/compare_runs.py runs/matrix/baseline runs/matrix/random_selection --output comparison.json
```

Ten arms are supported: baseline, random eligible selection, external consolidation only, training without exit, QA-only, decision-only, simple cold eviction, raw-experience decision SFT, unconsolidated external memory, and no memory.

Raw-experience decision SFT selects unconsolidated entries that already passed Writer provenance checks, uses their original observations/actions, and explicitly lowers support eligibility to one source task. It does not train indiscriminately on every raw trajectory. QA-only and decision-only change material and eligibility, so they are system ablations. Support-based and random selection share eligibility and compare the selection rule.

Generation does not execute experiments; the runner executes only explicitly selected arms. Shared-budget fingerprints cover models, environments, load, utility and limits. Inspect actual training tokens, updates, completed tasks and failure rates. Unequal use cannot be described as a pure selection effect under equal training. Seeds, stream lengths, change schedules and formal arm selection still need to be frozen.

## Failures, resources and recovery

Terminated training never publishes its candidate. Committed summaries and selected external memory remain. Unknown calls retain quota reservations so restarting does not make possibly consumed computation free. Temporary training payloads travel through bounded pipes, without creating an extra history available to future learning.

SQLite disables repeated cache spills during transactions and checkpoints WAL at transaction boundaries. Three times the main-database cap is reserved for main/WAL/SHM. Candidate adapters are reserved before training and checked against actual file sizes. Training monitors RSS and records CUDA allocator peaks. Sampling may miss brief overages, and allocator limits do not cover all CUDA or serving memory. Combine Slurm/cgroup limits, wall time, disk quotas and serving GPU/CPU adapter-cache limits for hard-budget acceptance.

Successful HTTP load/unload and `/models` changes establish API state, not immediate GPU cache release. Check the selected server version and bound resources such as `max_loras` and `max_cpu_loras`. See the [official vLLM LoRA documentation](https://docs.vllm.ai/en/latest/features/lora/); management endpoints belong in a controlled research service.

See [orphan archives and manual restoration](adapter_archive_recovery.md). Archives, audit logs and analysis reports are not learning inputs, but consume disk. Restored files must re-enter the budget if used again. Local directory state cannot determine whether a remote version has other users, so unknown advertised versions are preserved for deployment-owner review.

## Auto-Dreamer artifact check

At the recorded check, the [paper page](https://arxiv.org/abs/2605.20616) and [Artifact Details](https://arxiv.org/html/2605.20616v1#A3) did not yield a confirmed direct release of the authors' trained Consolidator weights and matching implementation. The paper lists OpenTinker and verl dependencies. This means a reusable artifact was not verified, not that none exists. The implementation remains explicitly a prompted region-rewriting prototype; it does not claim full reproduction or adopt the paper's model as MemTurn's formal selection.

## Acceptance and remaining deployment work

The handoff bundle's `validation/` directory contains logs, manifests and two isolated SFT records; exact counts are in `validation/summary.json`. Added checks cover long-stream deduplication, out-of-order recovery, rollback, exclusive ownership, process/descendant termination, output/memory limits, overload accounting, real loopback HTTP assembly and resume, deployment mismatches, unknown remote-candidate protection and ablations.

Cluster acceptance still needs actual model/tokenizer/loader/target compatibility, server weights and startup records, native task instances and verifier coverage, total RAM/VRAM peaks, concurrency and latency, frozen streams and budgets, and long-run statistics. Distributed training is outside the current single-learning-job implementation. Models requiring it need a separate extension and acceptance; the bundle does not establish distributed SFT support.

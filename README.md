# MemTurn

[English](README.md) | [简体中文](README.zh-CN.md)

**Memory Internalization for Continual Agents under Resource Constraints**

MemTurn studies learning across changing task streams with bounded external memory, a fixed-rank LoRA adapter, and explicit learning and service budgets.

**Status: executable research prototype.** The memory lifecycle and a tiny gradient-based fixture run locally. A separate Qwen3-1.7B backend supports local GPU integration checks. ScienceWorld and ALFWorld experiments, the trained Auto-Dreamer Consolidator and long-run comparisons remain pending. Engineering checks do not establish research advantages.

Current implementation: [mixed supervision, native environments and growth accounting](docs/implementation_2026-10-24.md).

Latest investigation: [learning and retrieval repairs](docs/learning_retrieval_repairs_2026-10-24.md),
including measured fixes and the remaining limits on parametric learning.

## Quick start

Python 3.10+ is sufficient for the core fixture, which uses only the standard library:

```text
python run_memturn.py demo --config configs/smoke.json
python -X utf8 -m unittest discover -s tests -t . -v
```

With PyTorch installed, add `--train-tiny` to exercise CPU gradients on a tiny action classifier. This classifier is separate from language-model SFT.

## Local GPU model

The local profile uses pinned [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) BF16 weights. The Windows script installs CUDA 12.8 PyTorch into `.venv-local` and model assets under `models/`:

```powershell
.\scripts\install_local.ps1
.\.venv-local\Scripts\python.exe run_memturn.py local-smoke --tasks 8
.\.venv-local\Scripts\python.exe run_memturn.py local-smoke --tasks 8 --operators fixture --train
```

The first run uses the model for the task, Writer and Consolidator. The second uses verified fixture operators to isolate real task-model inference and language-model LoRA SFT. Both use a small development environment and wait for background work between tasks. This does not measure uninterrupted service.

See [local model notes](docs/local_model.md) for hardware, measured results and reproduction.

A fixed API Writer can be selected independently with `--writer-profile`.
See [free-tier options and setup](docs/writer_api.md). The Groq pilot admitted
four memories and completed one local LoRA publication; Gemini remains untested.

For quota-aware exploratory streams, see [grouped-chain setup](docs/grouped_chain.md).
This mode plans up to 500 grouped synthetic tasks, records raw Writer candidates,
and stops at its token budget with a clean resume checkpoint.

## System design

Each task pins a model version and memory view, retrieves initially, and can request bounded additional retrieval. Background work writes grounded experience, consolidates a bounded region, and commits valid replacement memories.

A separate selector chooses at most one eligible consolidated entry per regular batch, preferring more verified distinct task-instance supports. The student cannot retrieve that entry, its parents or identified same-source aliases. Training continues the fixed-rank adapter; only a prepared candidate can update the published pointer and retire the selected memory version.

Capacity maintenance handles cold memories separately. It consolidates an initial fraction of the oldest entries within a region budget, then retains whole results ranked by verified task support within half the processed input tokens. The return count emerges from the budget. Capacity events do not train. Refused writes are recorded.

## Repository map

| Location | Responsibility |
| --- | --- |
| [docs](docs/README.md) | Architecture, setup, validation and literature |
| [configs](configs/README.md) | Offline, local and remote profiles |
| [scripts](scripts/README.md) | Isolated installation and pinned downloads |
| [src](src/README.md) | Replaceable memory, model, environment and training modules |
| [tests](tests/README.md) | Lifecycle, failure, concurrency and backend checks |

Start with [architecture](docs/architecture.md), [setup](docs/setup.md) and [validation status](docs/validation.md). Major directories have separate English and Chinese READMEs.

## Benchmark phase

Initial targets are **ScienceWorld and ALFWorld**; WebArena is deferred. Remote profiles follow Auto-Dreamer's task model `Qwen/Qwen3.5-9B` and Writer/Consolidator base `Qwen/Qwen3-14B`. The current prompted Consolidator is not the original trained checkpoint.

Remote hardware, model revisions, LoRA target modules, assets, task manifests and benchmark evidence verifiers need validation before formal runs:

```text
python run_memturn.py check-profile --profile configs/scienceworld.example.json
```

Incomplete templates stop before model execution. Task success alone does not certify every intermediate action as valid supervision.

## Reproduction and publication scope

Reports, audits, state databases and adapters are written to `runs/`. Weights, virtual environments, caches, credentials and generated outputs are excluded from Git. Pinned profiles and dependency records provide the setup path.

Only the `MemTurn` checkout is published; parent-workspace notes, papers and credentials remain separate. Clone with:

```text
git clone https://github.com/RuneTan/MemTurn-Memory-Internalization-for-Continual-Agents-under-Resource-Constraints.git MemTurn
cd MemTurn
```

Benchmark performance, long-run stability and novelty remain empirical questions. See the [primary literature](docs/literature.md). A software license has not yet been selected.

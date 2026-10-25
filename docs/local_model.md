# Local GPU integration

[English](local_model.md) | [简体中文](local_model.zh-CN.md)

Checked on 2026-10-23. These are development-environment checks, not benchmark
results or evidence that memory internalization improves task performance.

## Hardware and isolation

The local machine has an RTX 5080 Laptop GPU with 16,303 MiB VRAM. The previous
global PyTorch CUDA build lacked `sm_120` kernels. The project now has a separate
`.venv-local` with PyTorch 2.10.0+cu128. A real CUDA matrix operation passed and
the installed build includes `sm_120`. The global Python installation was kept.

The model is [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, loaded in BF16. This is a development
substitute for the larger remote models, not an Auto-Dreamer reproduction.
PyTorch wheels follow the [official CUDA 12.8 installation source](https://pytorch.org/get-started/previous-versions/).

| Item | Location or version |
| --- | --- |
| Python environment | `.venv-local/`, Python 3.10.6 |
| Weights and tokenizer | `models/qwen3-1.7b/` |
| Local profile | `configs/local_qwen3_1_7b.json` |
| Transformers | 5.2.0 |
| PEFT | 0.19.0 |
| Accelerate | 1.12.0 |
| Full dependency snapshot | `configs/local-windows.lock.txt` |
| Model backend | `src/memturn/models/local.py` |

Only the selected backend imports these dependencies. The core fixture still
runs with the standard library. Runtime loading uses local files; downloading is
an explicit, separate setup step with a pinned public model revision.

## Reproduce

From the checkout on Windows:

```powershell
.\scripts\install_local.ps1
.\.venv-local\Scripts\python.exe -X utf8 -m unittest discover -s tests -t . -v
.\.venv-local\Scripts\python.exe -X utf8 run_memturn.py local-smoke --tasks 8 --operators fixture --train
```

The fixture operators produce verifiable memories; task actions still come from
the real language model, and training uses the real Hugging Face SFT backend.
To test model-generated Writer/Consolidator output explicitly:

```powershell
.\.venv-local\Scripts\python.exe -X utf8 run_memturn.py local-smoke --tasks 8 --train
```

Each run creates a new ignored `runs/local_*` directory. Its `report.json` records
task outcomes, model usage, GPU peaks, package versions and event counts.
`audit.jsonl` includes generated responses, validation rejections and adapter
publication records. These records are not available through memory retrieval.

## Measured checks

| Configuration | Task success | Adapter publications | Peak CUDA allocated | Observation |
| --- | --- | --- | --- | --- |
| Local task model, fixture Writer/Consolidator, SFT enabled | 5/8 | 2 | 10.70 GiB | Real adapter continuation and publication completed; no background errors |
| Local model for task/Writer/Consolidator, SFT enabled | 8/8 | 0 | 4.06 GiB | All eight Writer candidates failed evidence checks, so no memory was admitted |
| Local task model, Groq GPT-OSS 120B Writer, fixture Consolidator, SFT enabled | 4/4 | 1 | 9.09 GiB | Four Writer memories admitted; one consolidation and publication; no background errors |

Local reports: `local_20261023T145345681297Z` and
`local_20261023T145602611706Z`. These directories are retained locally, not
published with weights or raw run data. The two short runs are integration
checks with different operators and state trajectories, not matched controls.
Their task-success difference must not be interpreted as a training effect.

The later Groq pilot is `local_20261023T151237196728Z`; its published adapter was
not used on later tasks in this four-task stream. See [Writer API notes](writer_api.md).

Each SFT job used rank 8, `q_proj`/`v_proj`, 1,605,632 trainable parameters and
two optimizer steps. The second published adapter continued the first. Finite
losses and changed LoRA parameters were checked; this does not demonstrate
learning quality. The second version was published at the end of the stream,
so this run does not establish its subsequent task performance.

## Writer limitations and fail-closed behavior

The initial model Writer confused feedback with action targets, source indices
and applicability labels. Source examples now support task/step references:
the program copies the original observation/action and independently validates
them. This removes the need for the model to reproduce long source strings;
it does not certify the model's conclusions.

The local 1.7B Writer still produced unsupported summaries or incorrect condition
labels. These candidates are rejected. Foreground task execution continues, but
without valid admitted memories there is no material for consolidation or SFT.
By agreement, further tuning of this small Writer is deferred. Fixture operators
remain available for engineering checks; larger remote operators need validation.
No stronger-model control has been run, so model capacity and prompt effects
have not been causally separated.

## Memory capacity is a provisional engineering choice

The local profile caps active memory at **8,000 task-tokenizer tokens**. This is
the sum of applicability and content across active entries. It is a starting
budget for integration, not an optimized value and not a conversion of VRAM.
The bank is external; a task receives at most 512 retrieved tokens, while one
consolidation region is at most 2,400 tokens and its candidate output at most
1,200 tokens. Capacity return is additionally limited to half the processed input.

The general default is 12,000 tokens. The original offline smoke profile uses
1,200 approximate lexical units, which are not directly comparable to Qwen
tokens. Sources and training samples have a separate 32 MiB cap; metadata,
snapshots and database pages also have separate accounting limits.

The eight-task runs did not reach the memory limit. Capacity behavior is covered
by regression tests, not by these short model runs. Formal experiments should
measure uncontrolled memory growth and choose task streams long enough to hit
the tested budgets. A possible sensitivity grid is 2k/8k/32k tokens under the
same budget for every method; it is a proposed experiment, not a validated setting.

## Remaining checks

The local model serializes generation and adapter switching; the two short runs
wait for maintenance between tasks. Neither measures uninterrupted service or
GPU contention. Real benchmark assets, evidence verifiers, original trained
Consolidator weights, remote LoRA compatibility and long-run matched-budget
comparisons remain pending.

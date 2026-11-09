# Development

[English](development.md) | [简体中文](development.zh-CN.md)

Integrating MemTurn means choosing a task environment and models, then connecting them to the task, experience-processing, and learning interfaces. Core scheduling and storage use Python 3.10+ and the standard library.

## Install and test

From the repository root:

```text
python -m pip install -e .
python -X utf8 -m unittest discover -s tests -t . -v
```

Tests use temporary databases and deterministic stand-ins, so they run offline. They cover retrieval, source checks, training inputs, USE and DISUSE, publication, and recovery. For process resource measurement, install the optional dependency with `python -m pip install -e ".[runtime]"`.

## Connect an application

| Prepare | Integration point |
| --- | --- |
| Tasks and environment | Assign unique task IDs and implement `reset`, `step`, and `close`; see wrappers in `tasks/` |
| Task model, Writer, and Consolidator | Provide `ChatClient.complete` for the prompt-based components, or implement the [component interfaces](components.md) |
| Learner | Implement `train`, `prepare`, and `retain` for fixed-rank LoRA training, loading, and saving |
| Token counting | Supply the model tokenizer to `TokenizerCounter`; use `OfflineCounter` for offline component tests |
| Runtime configuration | Create `Config`, `Store`, `Validator`, and `Runtime`, then execute tasks with `run_task` |

The ScienceWorld and ALFWorld wrappers require the corresponding environment dependencies and assets. For a separate environment process, configure `ProcessEnvironment` with a `worker_command` that handles `configure`, `reset`, `step`, and `close`. The model client handles transport, timeouts, and cancellation; the learner consumes the supplied `histories` and `memory_context` on every call.

Start integration with a short task stream and inspect memory writes, summary replacement, training inputs, and model loading. Then check failure and concurrency behavior: training failure preserves the current model and target memory; publication gives new tasks the new version while running tasks retain their original version. Check deployment resource limits and actual model token counts at the same time.

## Common settings

These `Config` values are development defaults. Adjust them to task lengths and resource budgets when preparing experiments.

| Setting | Default | Controls |
| --- | --- | --- |
| `batch_size` | 4 | Recorded task outcomes between USE cycles |
| `capacity_interval_batches` | 4 | USE batches between DISUSE cycles; an integer greater than 1 |
| `memory_token_cap` | 12000 | Total active external-memory tokens |
| `region_token_cap` | 4096 | Input region size for one Consolidator call |
| `output_token_cap` | 2048 | Memory-text output cap for one consolidation |
| `max_capacity_regions` | 2 | Maximum regions processed sequentially in one DISUSE cycle |
| `entry_token_cap` | 512 | Size limit for a memory selected for learning |
| `min_support_tasks` | 2 | Minimum distinct source tasks cited by a learning candidate |
| `max_examples_per_job` | 32 | Maximum training examples retained from one consolidation output |

Default USE/DISUSE intervals are 4/16 task outcomes, counting successes, failures, and rejections. USE runs first when both are due. A busy background worker coalesces overdue cycles and handles the currently due work. Read-only evaluation uses a separate read path, preserving maintenance counters and memory recency.

**A region is a group of memories: two regions can contain many entries.** DISUSE scans candidates in coldness order, filling `region_token_cap` with whole entries. Each region's retention budget is `min(floor(actual input tokens / 2), output_token_cap)`; the current design fixes `return_token_ratio` at `0.5`.

See [`config.py`](../src/memturn/config.py) for further storage, queue, retrieval, and version budgets. Model token counts depend on the supplied tokenizer.

## Compare configurations and inspect events

[`matrix.py`](../src/memturn/matrix.py) generates comparison configurations such as external consolidation alone, random learning-target selection, retaining memory after training, and using a single training-material type. Run these configurations under the same task stream and resource budgets, recording task utility and costs to compare component contributions.

Pass an `audit` callback to Runtime to collect writing, consolidation, material-filtering, training, and publication events. Hints such as `qa_text_overlap` help locate examples for review. See the [test guide](../tests/README.md) for regression-test entry points and coverage.

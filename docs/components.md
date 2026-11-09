# Components and interfaces

[English](components.md) | [简体中文](components.zh-CN.md)

The components below follow an experience from task execution to a model update. Paths are relative to `src/memturn/`.

| Component | Input and role | Implementation |
| --- | --- | --- |
| Task policy and environment | The policy chooses actions from observations, memories, and history; the environment executes them and returns feedback | `models/protocol.py`, `tasks/base.py` |
| Writer | Turns one task episode into a memory with source references | `models/agents.py` |
| Memory bank and retrieval | Stores episodes, memories, and versions; retrieves entries within a task's view | `memory/store.py`, `memory/retrieval.py` |
| Consolidator | Turns a region and its sources into summaries and training material | `models/agents.py` |
| Source checks and selection | Checks references and material; selects learning targets and cold regions | `memory/validation.py`, `memory/policies.py` |
| Learner | Trains on supplied material, prepares model versions, and retains those in use | `learning/base.py`, `learning/materials.py` |
| Runtime | Coordinates tasks, background maintenance, and model publication | `runtime/service.py` |

## Task execution and experience processing

The policy's `act(observation, memories, model_version, history)` returns an environment action or a memory-search request. Environments expose `reset(task)`, `step(action)`, and `close()`, with step feedback represented by `Transition`.

The Writer's `write(episode, batch)` returns a candidate memory, or `None` when there is nothing to record. The Consolidator's `consolidate(region, mode, batch, output_budget)` returns summaries and their training material. Prompt-based implementations live in `models/agents.py`; an injected `ChatClient.complete(messages, model_override)` connects them to the chosen model service.

`Validator(store)` checks memories and material against stored sources. [Sources and training material](provenance.md) explains how references are created and checked.

## Learning and publication

The learner uses one interface:

```text
train(examples, parent_version, *, histories, memory_context=()) -> Artifact
prepare(artifact) -> None
retain(versions) -> None
```

- `examples`: rule QA and decision demonstrations for the selected summary.
- `parent_version`: the model version to continue fine-tuning.
- `histories[(task_id, step)]`: the original task steps preceding that decision. A first-step decision has an empty tuple; QA-only training has an empty dictionary.
- `memory_context`: the tuple of external memories available to this training job, which may be empty.

`train` returns an `Artifact`, `prepare` makes that model version ready to use, and the runtime publishes it. `retain` receives the versions still in use so it can clean up older artifacts. The application's learner implements model loading, fixed-rank LoRA training, and saving.

## Assemble the components

Create `Store(path, config, counter)` and `Validator(store)`, then supply the Writer, Consolidator, and learner to `Runtime(store, writer, consolidator, validator, learner=...)`. Execute tasks with `run_task(task, environment, policy)`. At shutdown, close the Runtime before the Store.

Supporting modules include:

| Purpose | Modules |
| --- | --- |
| Data types, configuration, and token counting | `contracts.py`, `config.py`, `tokens.py` |
| Model-call budgets | `models/budget.py` |
| Adapter archives and recovery | `learning/archive.py` |
| Task arrivals, deadlines, and recovery records | `runtime/arrival.py`, `runtime/recovery.py` |
| Subprocess communication and resource measurement | `runtime/process.py`, `runtime/resources.py` |
| Environment integration and process bridging | `tasks/scienceworld.py`, `tasks/alfworld.py`, `tasks/process.py` |
| Comparison configurations | `matrix.py` |

See [Development](development.md) for installation, integration requirements, and common settings.

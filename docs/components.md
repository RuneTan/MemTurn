# Components and contracts

[English](components.md) | [简体中文](components.zh-CN.md)

| Module | Responsibility and boundary |
| --- | --- |
| `contracts.py` | Immutable task, episode, evidence, memory, snapshot and artifact records |
| `config.py`, `tokens.py` | Validated limits and injectable counters; lexical counting is not model-token accounting |
| `memory/store.py` | SQLite state, sources, versions, transactions, queues and recovery bookkeeping |
| `memory/retrieval.py` | Bounded search over the task's memory view |
| `memory/validation.py` | Provenance, replacement and training-material admission |
| `memory/policies.py` | One-entry selection, cold-region selection and half-budget packing |
| `models/agents.py` | Client-independent prompted Writer and regional Consolidator |
| `models/protocol.py` | Chat client contract, action formatting, policy and deadline contexts; no transport |
| `models/budget.py` | Token reservations and consumption reconciliation around an injected client |
| `learning/materials.py`, `learning/base.py` | Grounded QA/decision material and the learner lifecycle contract |
| `learning/archive.py` | Checksummed adapter archival and recovery utilities |
| `runtime/service.py` | Foreground/background coordination, publication, retirement and cleanup |
| `runtime/arrival.py`, `runtime/recovery.py` | Offered load, deadlines, directory leases and task-manifest identity |
| `runtime/process.py`, `runtime/resources.py` | Bounded subprocess communication and resource measurement |
| `tasks/base.py`, `tasks/process.py` | Environment/verifier contracts and process bridge |
| `tasks/verification.py`, `tasks/isolated.py`, `tasks/observed.py` | Replay checks, isolated verification and narrow observed-transition predicates |
| `tasks/scienceworld.py`, `tasks/alfworld.py` | Optional environment wrappers; assets and dependencies are not bundled |
| `matrix.py` | Pure configuration transformations for comparison arms |

## Supply these application components

- **Policy:** `act(observation, memories, model_version, history)` returns an environment action or a bounded memory-search request. `ChatPolicy` can use an injected `ChatClient.complete(messages, model_override)` implementation.
- **Writer:** `write(episode, batch)` returns a candidate memory or `None`. The prompted implementation uses public episode evidence.
- **Consolidator:** `consolidate(region, mode, batch, output_budget)` returns candidate replacement memories. Its bounded tool loop can inspect only allowed region/source evidence.
- **Verifier:** `verify_support(memory, episode, support)` and `verify_example(example, episode)` certify domain-specific evidence. Missing proof rejects admission; success on a task does not validate every step.
- **Learner:** `train(examples, parent_version)` returns an `Artifact`; `prepare(artifact)` ensures it can be used; `retain(live_versions)` releases only unused versions. Rich material support can additionally accept permitted memories and histories through the runtime's explicit capability branch. Applications implement actual optimization, fixed-rank continuation, persistence and loading.
- **Environment:** `reset(task)`, `step(action)` and `close()` exchange public observations and `Transition` records. Optional wrappers never install their dependencies automatically.

Construct `Store(path, config, counter)`, a `Validator(store, verifier)`, then `Runtime(store, writer, consolidator, validator, learner=...)`. Execute with `run_task(task, environment, policy)` and close runtime before store. No default backend is selected. Passing `learner=None` disables parameter learning; a read-only runtime uses no Writer, Consolidator or validator.

The complete method needs functional application components. An interface alone does not implement learning or prove internalization. See [development responsibilities](development.md).

# MemTurn

[English](README.md) | [简体中文](README.zh-CN.md)

**Bounded memory and selective internalization for continual agents.**

MemTurn is a model-independent engineering framework for an agent that learns from a changing task stream. It keeps external experience within explicit budgets, consolidates related experiences, prepares grounded supervision, and coordinates parameter publication with memory retirement.

This repository contains the core implementation, component contracts and regression tests. It does not ship a model, an API client, model-download scripts, experiment datasets or pilot results. An application supplies its own policy, experience operators, domain verifier and learner. The framework is not a ready-to-run benchmark service and does not establish a learning advantage.

## How it works

1. Pin the current model version and memory view for a task; retrieve before acting.
2. Write experience supported by that task's observations, actions and feedback.
3. Consolidate a bounded region across tasks into replacement memories and training material.
4. Select at most one eligible memory, prioritizing verified support from distinct tasks.
5. Train without exposing that memory or its identified source aliases to the student.
6. Prepare the new model version, then atomically publish its pointer and retire the selected memory version.

Capacity maintenance is separate: summarize a cold region and retain whole outputs within half its input token budget, without starting training. Publication is a successful engineering transition, not proof that the model mastered the experience.

## Explore the framework

| Entry | Contents |
| --- | --- |
| [Architecture](docs/architecture.md) | Data flow, permissions, lifecycle and resource limits |
| [Components and contracts](docs/components.md) | Concrete modules, inputs, outputs and extension points |
| [Development](docs/development.md) | Installation, offline tests and integration responsibilities |
| [Source map](src/README.md) | Implementation directories |
| [Literature](docs/literature.md) | Related primary sources |

The design assumes a frozen base model and a fixed-rank adapter supplied by the application. Writer and Consolidator behavior is fixed during a stream. No independent teacher, reinforcement learning, growing parameter capacity or mastery-gated retirement is built in.

Published without a software license (GitHub: None).

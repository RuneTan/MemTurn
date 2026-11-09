# MemTurn

[English](README.md) | [简体中文](README.zh-CN.md)

**Organize and learn from an agent's experience within fixed storage and compute budgets.**

An agent accumulates experience as it works through a stream of tasks. Which experiences should stay in memory, which deserve training compute, and how should the system make room for new ones? MemTurn connects external memory management with ongoing model updates to study these questions.

It manages experience at two cadences:

- **USE: organize recent experience and choose one summary to learn from.** A Consolidator turns new records and recently retrieved memories into summaries, rule questions and answers, and decision demonstrations. The system then selects a summary from the memory bank with broad source coverage and complete training material, fine-tunes the task model, and removes that external memory when the new version is published.
- **DISUSE: periodically compress less-used experience.** At a slower cadence, the same Consolidator reorganizes memories selected in coldness order. The system retains summaries within half the input's token count, making room for future experience.

Both paths run on fixed schedules. The task model uses a frozen base with a fixed-rank LoRA adapter; the Writer and Consolidator remain fixed throughout the task stream. The research question is whether this allocation of memory and learning budgets improves cumulative task utility and keeps earlier experience useful over time.

## Follow an experience

**Perform tasks with memory retrieval → record experience → produce summaries and training material → select a summary for fine-tuning → publish the model and remove the corresponding memory.**

By default, selection favors summaries citing more distinct source tasks. For example, among two eligible summaries drawn from three and two tasks respectively, the former takes priority. Each USE cycle trains on at most one summary; the others remain available for retrieval or later learning.

## Start here

The repository provides task scheduling, memory management, experience consolidation, training interfaces, and model version management. Connect a task environment and model backend to build a continual-learning experiment.

| What would you like to understand? | Read |
| --- | --- |
| How USE and DISUSE work together | [Architecture](docs/architecture.md) |
| Each component's inputs and outputs | [Components and interfaces](docs/components.md) |
| Where sources and training material come from | [Sources and training material](docs/provenance.md) |
| How to connect a model, configure the system, and run tests | [Development](docs/development.md) |
| The code and related research | [Source map](src/README.md) · [Literature](docs/literature.md) |

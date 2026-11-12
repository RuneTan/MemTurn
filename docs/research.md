# Research direction

[English](research.md) | [简体中文](research.zh-CN.md)

**Which experiences should an agent learn into its parameters, and how should it make room for new experience?** MemTurn studies this question in continuing task streams with fixed storage and compute budgets.

## The policy we study

USE organizes recent experience, prioritizes summaries by distinct source-task coverage, and combines rule questions with recorded decisions to update a fixed-size LoRA adapter. Publishing the update retires the selected external memory version. Slower DISUSE compresses cold memories. We study whether these choices improve long-term task utility and preserve the value of earlier experience.

USE first commits the new summaries in place of the selected region, then chooses at most one eligible summary from the whole bank, including previously retained summaries. If none qualifies, the summaries remain available for retrieval. The [architecture](architecture.md) and [sources and training material](provenance.md) pages explain this flow and its eligibility rules.

## How this relates to earlier work

The Consolidator's read-only inspection and region-replacement workflow draws on [Auto-Dreamer](https://arxiv.org/html/2605.20616v1). Our [`ChatConsolidator`](../src/memturn/models/agents.py) uses a fixed prompt and bounded tool calls, with the model supplied through the application. Auto-Dreamer trains its consolidator with GRPO. MemTurn extends the workflow to prepare rule-question and original-decision supervision alongside summaries; the surrounding policy selects the learning target and manages retirement.

[Dual-Layer Agentic Memory](https://arxiv.org/html/2608.22215v2#S3.SS4) couples parametric consolidation with external-memory removal after retrieval-free checks. MemTurn studies rules and recorded decisions in continuing interactive tasks, with retirement tied to model publication. The experiments below compare external-memory alternatives and MemTurn's design choices.

## Three linked experiments

The planned evaluation uses ScienceWorld and Database Exploration. The experiments share task splits, resource limits, and saved states where appropriate.

| Experiment | Question | Planned evidence |
| --- | --- | --- |
| 1. Overall performance | Does MemTurn improve service utility under fixed resources? | Compare full MemTurn, Auto-Dreamer, ReHarness, and external memory with cold-item eviction on matched task streams. Report deadline-aware utility, native task performance, and actual resource use. |
| 2. Value of accumulated experience | Does the accumulated state help with old and new situations? | Reuse Experiment 1 checkpoints. Compare each method's current state with its own initial state under the same subsequent feedback budget; both continue adapting. Report experience gain alongside absolute performance curves. |
| 3. Internal controls and ablations | Which design choices account for the effect? | Compare consolidation-only and raw-decision SFT, then examine source-coverage selection, the two supervision types, retirement, and cold-memory compression. Reuse full MemTurn as the reference. |

Auto-Dreamer retains its learned consolidator, and [ReHarness](https://arxiv.org/abs/2610.12668) retains its feedback-based reflection revision. Experiment 2 measures the combined value of model and memory state. Within Experiment 3, training-before/after and memory-visible/masked comparisons examine parameter-update effects; retaining versus retiring the trained memory examines the longer-term role of retirement.

Further reading: [literature index](literature.md). Implementation entry points: [components and interfaces](components.md).

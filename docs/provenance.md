# Sources and training material

[English](provenance.md) | [简体中文](provenance.zh-CN.md)

Each memory carries references identifying the task and step it came from. These references let the system revisit experience, construct training material, and count source coverage when selecting summaries.

## Where a reference comes from

The application assigns each episode a unique `task_id`, with steps numbered from 0. The Writer produces a memory from the current task and lists its source steps in `supports`, for example:

```json
[{"task_id": "task_a", "step": 2}, {"task_id": "task_a", "step": 5}]
```

When consolidating memories, the Consolidator declares which input memories contributed to the new summary and chooses references from the tasks associated with those parents. The summary can express the experience in new words; the references connect it to the original episodes.

The built-in `Validator(store)` checks the database directly: the task must remain available, the step number must be valid, and parents and sources must fall within the allowed scope. Summary content and applicability conditions must also be present. These are lightweight structural checks; summary quality is assessed through content review and subsequent task performance.

## How source coverage is counted

`source_coverage_count(memory)` counts distinct task IDs among valid references. For example, citing steps 2 and 5 of `task_a` and step 1 of `task_b` gives a source coverage count of **2**. Repeated references to the same task count once.

By default, USE ranks learning candidates by this count, and DISUSE uses it to prioritize summaries for retention. References are stored in `supports`; the minimum source-count setting is `min_support_tasks`, with a default of 2. Retrieval recency separately identifies recently used memories and orders cold memories.

## How the two material types are built

| Material | Input | Training target |
| --- | --- | --- |
| Rule QA | A question about the summary | The full summary or explicitly referenced content spans |
| Decision demonstration | The original observation at a source step, with its preceding task history | The action actually taken, accepted by the environment or marked as admissible |

The decoder copies the source observation and action for decision demonstrations. The validator checks the bound memory version, source scope, and fidelity to the original record. After deduplication, default mixed training requires at least one valid rule QA and at least one valid decision demonstration. The `qa_only` variant uses all retained valid rule QAs and requires at least one; both modes keep the existing sample and training budgets.

Material is processed sample by sample: retain valid samples, deduplicate identical examples and repeated content references, then apply the count limit in generation order. Summaries with incomplete material remain available as external memory. If a question contains its answer text, a `qa_text_overlap` event flags it for review; text overlap alone keeps the sample eligible.

## What the model reads during training

The learner receives training examples, original task history, and filtered external memory. To train on the selected experience without supplying it again as retrieved context, the external-memory input excludes the target summary, its parents, memories sharing its sources, and copies with the same content. History is rebuilt from original observations and actions, preserving the task experience; past retrieved text is handled through the external-memory channel.

See [Components and interfaces](components.md) for the complete training inputs.

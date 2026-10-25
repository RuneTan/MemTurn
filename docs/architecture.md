# Runtime architecture

[English](architecture.md) | [简体中文](architecture.zh-CN.md)

The research objective is cumulative utility over a specified changing task
stream under explicit memory, parameter, learning and service budgets. No
component's presence establishes a performance advantage.

```mermaid
flowchart TD
    Task[Task] --> Read[Initial / bounded additional retrieval]
    Bank[(Bounded SQLite memory)] --> Read
    Read --> Policy[Task model with pinned adapter]
    Policy --> Env[Independent task adapter]
    Env --> Trace[Public observations and feedback]
    Trace --> Writer[Background Writer]
    Writer --> Bank
    Bank --> Consolidator[Regular or capacity region consolidation]
    Consolidator --> Validate[Evidence and sample validation]
    Validate --> Bank
    Bank --> Select[Regular event: select at most one eligible entry]
    Select --> SFT[Candidate LoRA training]
    SFT --> Publish[Prepare adapter, then commit pointer and selected exit]
    Publish --> Bank
    Publish --> Policy
```

## Selection

An eligible entry is consolidated, within the entry-length budget, backed by at
least two different task instances, and has valid supervised examples. Support
counts are recomputed from currently available direct evidence; the same task
counts once. Highest count wins, with seeded random tie-breaking. No eligible
entry means no update. Unselected entries remain in external memory.

The selected entry, its parents and identified same-source aliases are excluded
from training retrieval. The language-model learner can receive remaining
retrieved memory. The tiny classifier uses only public observation features.
Independent semantic paraphrases are not automatically detected: this remains a
diagnostic obligation rather than a claimed perfect leakage detector.

## Policy and training contracts

Retrieval exchanges are separate from environment evidence steps. The policy
sees automatic retrieval, subsequent queries/results, errors and remaining calls.
Exhausted tool requests never become environment commands. The per-task decision
limit still bounds the loop. Memory-disabled evaluation removes both initial
retrieval and additional tool access.

Task adapters may expose a public applicability scope. The toy adapter's family
and version define incompatible rules, so it filters those conditions exactly;
unscoped environments retain lexical retrieval. This is not a universal filter
for transferable knowledge.

The current training profile shuffles and traverses selected samples once, with
one sample per optimizer update and no automatic accumulation growth. Historical
profiles with an explicit `steps` field retain their previous schedule. Jobs that
cannot finish the requested passes within token/deadline limits are rejected. Actual consumed
examples are recorded. Current examples retain the exact source state and action;
automatic source-state projection has been removed. Decision history stops before
the source action. Historical retrieval bodies/queries are not copied into SFT
inputs; permitted other memory is retrieved afresh before the static pass.
Updates and sample coverage still do not certify mastery;
the current publication policy does not contain a general behavioral mastery gate.

Static SFT retrieves allowed other memories while preparing its inputs. Its
forward/backward loop does not execute new model-generated tools. The Writer
keeps accepting new task evidence for an existing rule; there is no novelty-only
admission filter. Within a task, bounded retrieval history may remain in context;
new results do not have to replace old ones in a dedicated memory slot.

The default `training_material: mixed` requires a source-linked rule question
and at least one native decision example. Questions predict the selected content
as plain text; decisions predict environment-action JSON. Each example names its
memory version and covered content. Consolidation emits both kinds in the same
operation; no additional teacher or rewriting agent is introduced. The 32-example
event budget includes questions, which never increase task support. Invalid
training material is stripped from otherwise valid external memory. No silent
fallback to decision-only training occurs; that mode is explicit for the tiny
classifier and historical diagnostics.

## Capacity maintenance

The cold age is the foreground batch minus the last real task-access anchor.
Writing, inspecting, training or rewriting does not refresh that anchor. New
summaries inherit the latest anchor of their contributing parents.

Take the oldest initial fraction `m`, subject to a bounded input region. Generate
validated replacements, rank by distinct task support, and greedily retain whole
entries fitting at most half of the actual processed region's tokens. Oversized
entries are skipped. The resulting count `k` is measured, not configured. A
capacity event never trains the task model.

The trigger and target occupancy ratios leave headroom. A bounded number of
regions can be handled per event; unsuccessful compaction leaves the old region
intact. If admission still exceeds a limit, new experience is refused and that
event is recorded. No claim is made that every stream can be losslessly accepted.

## Publication and recovery

External consolidation commits first. Model publication is a later transaction,
after the candidate has been prepared for inference. Only the selected memory
version exits with the new model pointer. Training failure preserves the committed
external bank. Existing task leases retain their original model and memory view.

SQLite stores the current state and bounded job/results records. Pending writer
episodes can be requeued at startup. Writer insertion and its source-stage marker
commit together, so a restart after insertion does not run that Writer again.
Uncommitted training jobs are cancelled;
their candidates never become the active model merely because files exist.
The formal remote runner also persists a frozen task manifest, contiguous completion
watermark and bounded out-of-order window, so completed stream positions are not
counted again when recent results rotate. Earlier development entry points retain
only bounded result history and should not be treated as the same recovery protocol.
Neither unlimited arbitrary-ID replay protection nor exactly-once external-world
actions are claimed. See [cluster handoff](cluster_handoff.md).

## Accounting limits

The current implementation limits active text, evidence/sample data, metadata,
snapshots, queues, simulator steps, model versions and database storage. It also
reserves workspace, monitors training-process memory, enforces process deadlines,
persists compute quotas and archives orphan candidates before cleanup.

These software controls do not prove a deployment-wide hard memory bound.
Inference caches, remote serving caches and peaks still need measurement and
cluster quotas. See [cluster handoff](cluster_handoff.md) for details and limits.

The default lexical counter and retriever are development implementations. Remote
profiles use the task tokenizer; retriever choice and revisions must be frozen
for matched-budget comparisons. Audits are not returned through memory tools.

`stats.growth` records cumulative admitted tokens and net releases from regular
consolidation, internalization and capacity maintenance, plus an initial baseline
for existing stores. Counters update in the same transaction as memory changes.
They support growth analysis; no adaptive scheduler has been activated.

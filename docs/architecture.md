# Architecture

[English](architecture.md) | [简体中文](architecture.zh-CN.md)

The foreground executes tasks against pinned model and memory versions. The background writes, consolidates and learns. A task already in flight keeps its own view when later publications occur.

```mermaid
flowchart LR
  T[Task and observation] --> R[Bounded retrieval]
  R --> A[Policy and environment]
  A --> W[Grounded Writer]
  W --> B[External bank]
  B --> C[Regional Consolidator]
  C --> B
  C --> S[Eligible memory selection]
  S --> L[Application learner]
  L --> P[Prepare and publish]
  P --> B
```

## Two maintenance paths

Regular batches default to four completed source tasks. The Consolidator reads a bounded region and its available source evidence, and emits a replacement set. Source verification and atomic replacement precede learning. At most one eligible entry with at least two distinct supporting tasks is selected. Default mixed supervision combines a rule QA with original source decisions; the learner must honor the selected material, exclusions and configured training limits.

A successful prepared artifact allows one transaction to change the published pointer and retire the selected entry version. A failed learning job preserves the committed external summaries. Retirement is not gated on a mastery test.

Capacity pressure selects cold entries separately. Outputs are ranked by verified support and packed whole within half the processed input tokens; the number retained is determined by the budget. This event never trains.

## Evidence and permissions

The policy reads retrieved memory; it does not directly write the bank. The Writer sees one public episode. The Consolidator may inspect the selected region and accessible sources. The verifier checks source links and domain validity independently. Generated text does not certify itself. Hidden answers and evaluation metadata are not public experience.

Training excludes the target, its parents and identified same-source aliases. Read-only evaluation does not update memory access counters or restore pending learning jobs. Audit logs are analysis outputs, not a retrievable experience store.

## Budgets and recovery

`Config`, `Store` and `Runtime` bound active memory, retained sources, queues, retrieval, supervision and version state. SQLite transactions protect replacement and publication. Recovery records retain pending writes, stream position and a bounded completion window. Finite queues and task deadlines support offered-load accounting.

Applications must still enforce deployment-wide RAM/GPU/storage limits and cancellation in their backends. The arrival driver is not a fixed-horizon evaluator; whole-run cutoff, parent-chain settlement, checkpoint branching and scientific evaluation remain separate integration work. No infinite deduplication or exactly-once external action guarantee is claimed.

See [component contracts](components.md) and [development](development.md).

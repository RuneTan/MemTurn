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

Cold-memory maintenance selects cold entries on its slower fixed schedule, regardless of bank occupancy. Outputs are ranked by verified support and packed whole within half the processed input tokens; the number retained is determined by the budget. This event never trains.

## Evidence and permissions

The policy reads retrieved memory; it does not directly write the bank. The Writer sees one public episode. The Consolidator may inspect the selected region and accessible sources. The verifier checks source links and domain validity independently. Generated text does not certify itself. Hidden answers and evaluation metadata are not public experience.

Training excludes the target, its parents and identified same-source aliases. Read-only evaluation does not update memory access counters or restore pending learning jobs. Audit logs are analysis outputs, not a retrievable experience store.

## Budgets and recovery

`Config`, `Store` and `Runtime` bound active memory, retained sources, queues, retrieval, supervision and version state. SQLite transactions protect replacement and publication. Recovery records retain pending writes, stream position and a bounded completion window. Finite queues and task deadlines support offered-load accounting.

Applications must still enforce deployment-wide RAM/GPU/storage limits and cancellation in their backends. The arrival driver is not a fixed-horizon evaluator; whole-run cutoff, parent-chain settlement, checkpoint branching and scientific evaluation remain separate integration work. No infinite deduplication or exactly-once external action guarantee is claimed.

See [component contracts](components.md) and [development](development.md).

## Maintenance schedule

`Config.batch_size` controls the regular use/internalization interval in accounted source-task outcomes (default 4). `Config.capacity_interval_batches` is an integer greater than 1 (default 4); cold-memory consolidation without internalization runs every `batch_size * capacity_interval_batches` outcomes (default 16). Successful, failed and rejected outcomes advance the existing completion counter; read-only probes do not. The same Consolidator is used, with no training on the cold path. Coincident schedules run regular maintenance first, then cold maintenance.

Each cold event processes up to `max_capacity_regions` regions (default 2), even below the memory limit. Empty/failed attempts wait for the next window. Busy workers coalesce overdue windows; persistent attempt markers prevent repeated claims on restart. A claimed attempt interrupted by a crash is not guaranteed to replay. The hard memory cap still rejects oversized writes, without triggering extra consolidation. Occupancy-based `capacity_trigger_ratio` and `capacity_target_ratio` are removed; old configurations must replace them with `capacity_interval_batches`. These are development defaults, not tuned research settings.

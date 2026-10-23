# Runtime

[English](README.md) | [简体中文](README.zh-CN.md)

`Runtime.run_task` pins a model version and bounded memory view, performs initial and bounded additional retrieval, and records task outcomes. One bounded worker handles writing, consolidation and optional training.

Memory replacement commits first. A later publication transaction updates the model pointer and removes only the selected memory version. Failed training preserves the external bank. Capacity jobs do not train.

Foreground bookkeeping uses a separate short lock from maintenance. Cold commits reject changed access timestamps. Restart recovery requeues pending Writer sources and cancels uncommitted model jobs; Writer insertion and its completion marker commit together.

Command-line checks wait between tasks. Concurrent orchestration tests are distinct from real service-latency measurements; the local GPU backend serializes generation calls.

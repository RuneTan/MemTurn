# Runtime

[English](README.md) | [简体中文](README.zh-CN.md)

Retrieval calls and results have a separate policy transcript. Search errors are
returned to the policy and never sent to the environment as fabricated commands.
Environment evidence step indices remain unchanged. No-memory evaluation disables
both initial retrieval and additional searches.

`Runtime.run_task` pins a model version and bounded memory view, performs initial and bounded additional retrieval, and records task outcomes. One bounded worker handles writing, consolidation and optional training.

Memory replacement commits first. A later publication transaction updates the model pointer and removes only the selected memory version. Failed training preserves the external bank. Capacity jobs do not train.

Foreground bookkeeping uses a separate short lock from maintenance. Cold commits reject changed access timestamps. Restart recovery requeues pending Writer sources and cancels uncommitted model jobs; Writer insertion and its completion marker commit together.

Command-line checks wait between tasks. Concurrent orchestration tests are distinct from real service-latency measurements; the local GPU backend serializes generation calls.

Episodes persist actual initial/additional memory exposures separately from native steps. Public operator inputs exclude native private manifests/condition labels. Static SFT receives original environment history plus freshly retrieved permitted memory, without historical retrieval bodies or queries.

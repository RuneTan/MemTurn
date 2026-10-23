# Package map

[English](README.md) | [简体中文](README.zh-CN.md)

`contracts.py` defines immutable task, memory, evidence and release records. `config.py` defines budgets and rejects unknown settings. `tokens.py` separates the fixture counter from model tokenization.

| Directory | Responsibility |
| --- | --- |
| [memory](memory/README.md) | State, provenance, retrieval and selection |
| [tasks](tasks/README.md) | Environments and evidence verification |
| [models](models/README.md) | HTTP and local inference backends |
| [learning](learning/README.md) | Adapter training and artifact checks |
| [runtime](runtime/README.md) | Foreground sessions and background publication |

`cli.py` provides commands. `experiments.py` assembles remote profiles; `local_experiment.py` assembles local GPU checks. These assembly modules do not change the memory policy.

`long_chain.py` runs quota-aware grouped exploratory streams; `chain_support.py`
supplies bounded Writer retries and no-feedback held-out probes. Observer logs
are separate from the agent's bounded memory.

# Memory

[English](README.md) | [简体中文](README.zh-CN.md)

`store.py` persists the active bank, sources, bounded job/result records and model pointer in SQLite. Tasks hold immutable snapshots without keeping a transaction open during model calls.

`validation.py` delegates evidence and target checks to an independent environment verifier. Support uses verified unique task IDs and declared contributing parents. `policies.py` selects at most one eligible entry and packs whole capacity outputs into half the processed input tokens.

`retrieval.py` is a replaceable lexical retriever. `consolidation.py` contains deterministic development operators, separate from model-based operators.

Text, sources/samples, metadata, snapshots and database pages have separate limits. These limits do not bound complete process RSS or remote caches. Audits are not retrievable memory.

An environment may provide an explicit public applicability scope to retrieval.
The toy adapter excludes incompatible family/version rules; unscoped environments
keep lexical retrieval. This is not a general semantic relevance solution.

`stats.growth` separates admitted tokens from regular-consolidation, internalization and capacity releases, using transactional cumulative counters. It measures growth without activating adaptive scheduling. Mixed training validation checks ownership, covered content, typed targets and independent source support; invalid training material does not discard otherwise valid external memory.

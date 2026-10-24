# Task adapters

[English](README.md) | [简体中文](README.zh-CN.md)

Adapters implement `reset(task)`, `step(action)` and `close()`. A separate verifier checks whether sources support a memory and whether a target matches recorded evidence.

`toy.py` contains two stateful development families with changing versions, a visible action vocabulary and corrective feedback. The task model sees public observations, not the verifier's answer table.

Its optional `retrieval_scope(observation)` exposes public family/version
compatibility. Historical toy projection remains available only for explicitly named legacy
diagnostics; the current Validator rejects projected training inputs.

`scienceworld.py` and `alfworld.py` isolate optional simulator dependencies. They support original native environments through the isolated process adapter; domain evidence verification remains pending. `webarena.py` is a deferred stub.

`verification.py` rejects uncertified benchmark evidence. Task success alone does not certify all intermediate steps. Environment assets and credentials are excluded from Git.

`process.py` separates native simulator dependencies from the model process. `NativeReplayVerifier` checks replayed observations, feedback, native evidence, reward and success, with a bounded cache. Summary support additionally requires a separate domain predicate; admissibility and final success do not certify arbitrary free-text conclusions. See [implementation status](../../../docs/implementation_2026-10-24.md).

# Task adapters

[English](README.md) | [简体中文](README.zh-CN.md)

Adapters implement `reset(task)`, `step(action)` and `close()`. A separate verifier checks whether sources support a memory and whether a target matches recorded evidence.

`toy.py` contains two stateful development families with changing versions, a visible action vocabulary and corrective feedback. The task model sees public observations, not the verifier's answer table.

Its optional `retrieval_scope(observation)` exposes public family/version
compatibility. Its verifier can replay-check a later accepted preparation action
on the recorded initial input, with distinct input/evidence step indices. This
projection must not be assumed valid in other environments.

`scienceworld.py` and `alfworld.py` isolate optional simulator dependencies. Their wrapper contracts are tested with fake environments; actual assets and benchmark verifiers remain pending. `webarena.py` is a deferred stub.

`verification.py` rejects uncertified benchmark evidence. Task success alone does not certify all intermediate steps. Environment assets and credentials are excluded from Git.

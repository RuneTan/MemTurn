# Learning

[English](README.md) | [简体中文](README.zh-CN.md)

`base.py` defines candidate training, inference preparation and version retirement. `tiny_lora.py` tests CPU gradients on a small action classifier with a frozen base.

`huggingface.py` performs optional language-model SFT on verified prompt/target records. It shares the inference prompt and action schema, masks prompt labels, rejects unstable or oversized samples, continues the fixed-rank adapter and creates a new optimizer per job. Only LoRA parameters may train.

Public source history and permitted external memory are supplied after runtime exclusions. No teacher is required. Artifacts record checksums and training metrics; local and vLLM publication backends are independent.

See [validation](../../../docs/validation.md) for actual model checks and pending remote compatibility.

The current profile uses one shuffled pass (`epochs: 1`) and accumulation one:
each example produces one optimizer update. There is no fixed two-update cap or
automatic increase in accumulation in epoch mode. Token/deadline limits still
apply: incomplete passes prevent publication and memory exit. Explicit `steps`
settings retain the legacy schedule for historical profiles and diagnostics.
Artifacts record consumed source/input steps and hashes.
The toy adapter can replay-verify an initial-state input; other domains require
their own verifier. A successful optimizer update does not certify mastery.

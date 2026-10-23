# Learning

[English](README.md) | [简体中文](README.zh-CN.md)

`base.py` defines candidate training, inference preparation and version retirement. `tiny_lora.py` tests CPU gradients on a small action classifier with a frozen base.

`huggingface.py` performs optional language-model SFT on verified prompt/target records. It shares the inference prompt and action schema, masks prompt labels, rejects unstable or oversized samples, continues the fixed-rank adapter and creates a new optimizer per job. Only LoRA parameters may train.

Public source history and permitted external memory are supplied after runtime exclusions. No teacher is required. Artifacts record checksums and training metrics; local and vLLM publication backends are independent.

See [validation](../../../docs/validation.md) for actual model checks and pending remote compatibility.

The shuffled schedule covers every selected example before repeating. Gradient
accumulation grows when needed to cover the data within the configured optimizer
steps. Token/deadline limits still apply: incomplete coverage prevents publication
and memory exit. Artifacts record consumed source/input steps and hashes.
The toy adapter can replay-verify an initial-state input; other domains require
their own verifier. A successful optimizer update does not certify mastery.

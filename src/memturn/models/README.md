# Model backends

[English](README.md) | [简体中文](README.zh-CN.md)

`http.py` provides the remote chat client, shared task prompt format and JSON action schema. Each remote role has independent endpoint/model settings and named environment variables for keys.

Task prompts expose completed retrievals and remaining availability, omit search
when unavailable, and describe actual action/query fields without executable
placeholder strings. The same format is used for SFT and inference.

`local.py` loads local Transformers weights, checks a real CUDA kernel, and supplies the same `complete()` interface. It serializes generation and adapter changes, uses immutable adapter names, and keeps fixed Writer/Consolidator calls on the base model. Loading an adapter does not merge it into base weights.

`agents.py` supplies the prompted Writer and bounded read-only Consolidator tool loop. Independent evidence checks gate their outputs. This is not the trained Auto-Dreamer checkpoint.

No credentials are read from parent files. Local weights and deployment-specific serving code stay separate from memory management.

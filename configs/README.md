# Configuration

[English](README.md) | [简体中文](README.zh-CN.md)

`smoke.json` runs the standard-library fixture. `local_qwen3_1_7b.json` pins the local GPU model, tokenizer and SFT settings. The ScienceWorld and ALFWorld examples are incomplete remote templates.

```text
python run_memturn.py check-profile --profile configs/scienceworld.example.json
```

Role endpoints and keys are read from environment variables; keys never belong in JSON. Remote templates require verified revisions, target modules, task manifests and an evidence verifier. Their initial arrival policy waits for background work.

Budgets are defined in `src/memturn/config.py`. Capacity returns use a fixed 0.5 token ratio; `cold_fraction` controls the initial cold selection, then `region_token_cap` limits actual processing. The return count is not configured.

The local 8,000-token bank is a provisional integration budget, not a VRAM-derived
or experimentally optimized setting. Only up to 512 tokens are retrieved per
query. Sources/samples have a separate 32 MiB limit. The short GPU runs do not
exercise saturation; see [budget rationale](../docs/local_model.md).

`writer.groq.example.json` and `writer.gemini.example.json` configure an optional
HTTP Writer with the local task model. Keys are read only from the named process
environment variable. Provider access has not been exercised; see [API notes](../docs/writer_api.md).

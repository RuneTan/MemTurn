# Setup scripts

[English](README.md) | [简体中文](README.zh-CN.md)

`install_local.ps1` creates `.venv-local`, installs pinned Windows CUDA dependencies and downloads the local model. It checks command failures and does not alter global Python.

`download_local_model.py` reads the local profile and downloads public assets at an immutable revision under `models/`. It does not use a saved Hugging Face token.

Run from the checkout:

```powershell
.\scripts\install_local.ps1
```

Both environment and weights are ignored by Git. Measured setup details are in [local model notes](../docs/local_model.md).

`summarize_chain.py` reads exploratory run artifacts and produces group reports,
Writer review examples and Matplotlib PNG/SVG figures without model calls.

`evaluate_chain.py` runs local-only, complete held-out task episodes after a
chain has paused. It compares frozen/current adapters and absent/current memory,
with no Writer calls, training or learning-state changes.

`validate_repairs.py` runs bounded local repair diagnostics against a frozen Git
revision: fixed-bank retrieval ablations, replay-verified SFT data, actual sample
coverage and held-out first actions. It writes to a new run directory and never
resumes the source experiment or contacts Groq. Training-only calibration flags
can vary the fixed dataset and explicit step/token budgets.
`--training-epochs 1` selects the current complete-pass schedule when checking an
old run; it is mutually exclusive with `--training-steps`.

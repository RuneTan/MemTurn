# Setup scripts

[English](README.md) | [简体中文](README.zh-CN.md)

`build_handoff.py` creates an allowlisted collaboration bundle with file hashes and a ZIP. Use `--documents` to select the directory containing the three design documents and optional experiment manual, and `--summary` for the recorded validation summary. It supports both the reorganized `docs` layout and the earlier layout. Document subdirectories containing API keys, manuscript drafts or PDFs are not copied. The evidence paths refer to existing local validation artifacts; a fresh clone alone does not contain them. `verify_handoff.py` checks a completed bundle against its manifest.

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
revision: fixed-bank retrieval ablations, source-state SFT data, actual sample
coverage and held-out first actions. It writes to a new run directory and never
resumes the source experiment or contacts Groq. Training-only calibration flags
can vary the fixed dataset and explicit step/token budgets.
`--training-epochs 1` selects the current complete-pass schedule when checking an
old run; it is mutually exclusive with `--training-steps`.

`install_native_linux.py` installs pinned original simulators and a checksum-verified portable JRE inside the project. `prepare_native_checks.py` prepares official assets and explicit native interface manifests. `native_worker.py` exposes only reset/step/close through a separate process; `check_native.py` records original-environment interface checks.

`check_mixed_sft.py` checks actual local LM training with typed QA/decision targets, continued LoRA updates and adapter loading. Its QA is a diagnostic fixture; it does not certify native memory learning or retire memory. See [implementation status](../docs/implementation_2026-10-24.md).

`validate_repairs.py --legacy-projection` explicitly reproduces the retired toy projection. Without that flag it keeps source states. This historical action-only diagnostic is separate from the current mixed protocol.

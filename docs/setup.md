# Development and remote setup

[English](setup.md) | [简体中文](setup.zh-CN.md)

## Local checks

Python 3.10+ is sufficient for the memory-only fixture; it uses the standard
library. From this checkout:

```text
python run_memturn.py doctor
python run_memturn.py demo --config configs/smoke.json
python -m unittest discover -s tests -t . -v
```

If a compatible PyTorch is already available, add `--train-tiny` to the demo.
This does not download Qwen. All outputs go into a fresh ignored `runs/` directory.
The local global Python installation is not modified by these commands.

To install the package into a project environment, create that environment and
run `pip install -e .`. The `train`, `scienceworld` and `alfworld` extras are
optional compatibility ranges, not a validated remote lockfile.

## First benchmark phase

For isolated Qwen3-1.7B inference and language-model SFT on the laptop, see the
[local model instructions and measured limitations](local_model.md). This path
does not require benchmark assets or a remote server.

The agreed first environments are ScienceWorld and ALFWorld. WebArena is deferred.
The formal task, Writer and Consolidator models are pending collaborator discussion.
The remote templates intentionally leave model identities unresolved.
Auto-Dreamer's Consolidator was trained from Qwen3-14B. A base Qwen3-14B prompted
operator is only a development variant, not that trained checkpoint.
See [Auto-Dreamer section 5.1](https://arxiv.org/html/2605.20616v1).

Start with:

```text
python run_memturn.py check-profile --profile configs/scienceworld.example.json
```

This reports unresolved fields without connecting, downloading weights or
printing key values. Complete a copied profile after establishing:

1. GPU models/count/VRAM, Linux and Python versions, storage paths and compute budget.
2. Separate role endpoints and pinned model/tokenizer revisions. Use the variable
   names in `.env.example`; keys stay outside the repository.
3. The actual Consolidator checkpoint, or an explicitly named prompted variant.
4. Verified LoRA target modules and task-model loader. Qwen3.5 is a multimodal
   architecture; do not silently assume `AutoModelForCausalLM` works.
5. Training and inference access to the same bounded adapter directory. vLLM
   publication uses immutable names and its dynamic LoRA endpoints. The service
   must actually support LoRA for the selected model and loader combination;
   this is not locally validated. Follow the deployment requirements in the
   [vLLM documentation](https://docs.vllm.ai/en/latest/features/lora/).
6. Environment assets and a fixed JSONL task manifest with unique IDs, families,
   versions, seeds, step limits and environment-specific metadata.
7. A benchmark-specific `EvidenceVerifier`, including checked source support and
   supervised targets. `verification.py` intentionally rejects unverified claims;
   a successful simulator score alone does not certify every intermediate action.

ScienceWorld needs Java and its environment package. Pin its version: the
[official README](https://github.com/allenai/ScienceWorld) documents changes that
can affect trajectories. ALFWorld needs its text environment, game assets and
YAML configuration; see its [official setup](https://github.com/alfworld/alfworld).
Both are installed in a separate project Linux environment; see the current
[implementation record](implementation_2026-10-24.md). This does not complete the
domain verifier or remote deployment requirements above.

After those checks, the real entry point is:

```text
python run_memturn.py run --profile configs/your_verified_profile.json --output runs/your_new_run
```

The provided profiles intentionally contain unresolved fields and cannot launch
formal experiments unchanged. They now include bounded periodic arrivals and explicit `--resume`, with an immutable
manifest and bounded out-of-order completion window. A narrow observed-transition
verifier is available; free-form semantic claims remain gated. See
[cluster handoff](cluster_handoff.md) for process deadlines, quotas, deployment
attestation, the matrix driver and remaining cluster acceptance. Distributed SFT
and measured serving SLAs are not established by the offline fixture.

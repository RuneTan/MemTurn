# Validation status

Updated: 2026-10-23. This is implementation evidence, not paper results.

Latest isolated-environment regression run: **31 tests passed**. All 12 project
README locations have separate English and Chinese files with language links.

## Checked locally

- Source-checkout CLI and 24-task memory-only / tiny-LoRA fixtures: task execution,
  initial retrieval, writing, six regular consolidations, bounded state and reports.
  The latest tiny run completed six adapter publications with no background errors;
  these are engineering checks on a simple environment, not comparative results.
- Real CPU LoRA gradients on the tiny action classifier, frozen base weights,
  continued updates, saved adapter reload and retirement.
- Unit/integration checks for provenance deduplication, fabricated evidence,
  one-entry selection, half-budget packing, write rollback, cold-age inheritance,
  stale-access rejection, snapshot isolation, training failure, selected-version
  exit, model-pointer recovery, pending-Writer recovery without duplicate insertion,
  fail-closed remote profiles, and foreground work during blocked training.
- Chat request contracts with mocks; assistant-only SFT masks with a tokenizer
  fixture, including the shared inference action schema; ALFWorld and ScienceWorld
  wrapper contracts with fake simulators.

The exact final test count and run report paths are recorded in the delivery
message and the ignored run artifacts. Test fixtures do not stand in for live
model or environment integration.

Local environment: Windows, Python 3.10.6, PyTorch 2.14.0+cu126. The installed CUDA
build reports no kernels for the local RTX 5080 Laptop GPU, so the tiny check uses
CPU. No global package changes or remote deployments were performed.

An isolated `.venv-local` now supplies PyTorch 2.10.0+cu128 with working `sm_120`
kernels. Qwen3-1.7B GPU inference and two real language-model LoRA publications
were executed. The fixture-operator run succeeded on 5/8 tasks, while the all-model
operator run admitted no memories because its Writer candidates failed evidence
checks. Full measurements and their limitations are in [local model notes](local_model.md).

The Groq GPT-OSS 120B Writer was subsequently connected to the local task model
with fixture consolidation: all four tasks completed, four Writer memories were
admitted, and one real LoRA publication completed without background errors.
The run used 4,809 Writer tokens and peaked at 9.09 GiB CUDA allocation. This
verifies the mixed backend path, not a measured learning advantage.

## Not yet validated

- Remote Qwen3.5-9B loading, chat templates, LoRA target modules and vLLM adapter
  hot loading. Local Qwen3-1.7B has been checked separately.
- Hugging Face SFT against the actual remote target models. The complete backend
  has been exercised on local Qwen3-1.7B, not on remote Qwen3.5-9B.
- Installed ALFWorld/ScienceWorld assets, fixed task manifests and replay/domain
  verifiers. Automatic benchmark sample certification remains open and is gated.
- An actual trained Auto-Dreamer Consolidator checkpoint and faithful reproduction.
- Long-run resource/service budgets, inference contention, remote crash recovery,
  quantitative advantage over baselines, and any contribution claim.

The model, task and validation boundaries are explicit so these checks can be
completed without rewriting memory management. Each integration must update this
document and the relevant directory README.

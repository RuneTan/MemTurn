# Validation status

[English](validation.md) | [简体中文](validation.zh-CN.md)

Status reviewed: 2026-10-29. The engineering measurements below were recorded on 2026-10-24; they are implementation evidence, not paper results.

The [new experiment plan](experiment_protocol.md) adds three-environment evaluation, fixed-horizon utility, valid old-task retention and adaptation at different stream ages. Continual Learning Bench/MemoryArena bridges, a complete parent-request ledger with cutoff handling, consistent checkpoint export, and isolated learning branches remain unimplemented. Existing arrival, timeout and read-only evaluation support does not establish this full protocol. No new formal experiment results accompany this documentation update.

Latest delivery regression: **132/132 passed on Windows**; **129 passed and 3 optional-dependency skips on Linux**. The local HTTP/process fixture exercises the remote assembly and exact-once result accounting across resume. Two real isolated Qwen3-1.7B SFT jobs each covered all three fixture examples, with three updates and continued LoRA weights. These are engineering checks, not native task gains or a cluster SLA. See the [cluster handoff](cluster_handoff.md); the earlier [audit](audit_repairs_2026-10-24.md) preserves its historical validation counts.

Grouped-chain checks additionally cover raw HTTP response auditing, daily-limit
classification, a new allowance on explicit resume, source requeue without task
double-counting, and held-out probes that leave learning state unchanged.

Training/retrieval repairs additionally check complete shuffled sample coverage,
fail-closed token budgets, unmodified source-state inputs, optional applicability
scope, visible tool exchanges, memory-disabled evaluation and non-executable
prompt placeholders. Fixed-bank local diagnostics reproduced 4/8 completion with
the original implementation, 6/8 with scope alone, and 8/8 with the full tool
protocol. An eight-task real-LM/LoRA fixture run reached 8/8 after placeholder
removal, with two publications and no background errors. Learning remains
incomplete: a fixed balanced-data calibration reached 25% first-action accuracy
with two steps and 50% with 32. See [investigation and limits](learning_retrieval_repairs_2026-10-24.md).

The subsequently confirmed default uses one complete sample pass (`epochs: 1`),
one sample per update and accumulation one. Fixed-step profiles remain supported
for historical reproduction. An eight-task real-LM/LoRA check completed all eight
with two publications and no background errors; `epoch_schedule_check.json` in
`runs/static_sft_epoch1_20261024/` records actual sample/update counts. This does
not establish that a single pass is sufficient for internalization.

## Current mixed-supervision check

The current typed QA/action path, original-environment boundary and growth
accounting are documented in [implementation status](implementation_2026-10-24.md).
Two real rank-8 Qwen3-1.7B jobs consumed all three mixed examples each. The QA
probe originally returned action JSON, but an audit found that the probe itself
incorrectly requested action output. That result is not evidence of QA failure.
The corrected probe returns prose, which still does not establish internalization.
No memory was retired by either interface fixture. Earlier toy measurements
below are historical and do not validate the current native protocol.

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

Initial setup snapshot: Windows, Python 3.10.6, PyTorch 2.14.0+cu126. The installed CUDA
build reports no kernels for the local RTX 5080 Laptop GPU, so the tiny check uses
CPU. No global package changes or remote deployments were performed.

An isolated `.venv-local` now supplies PyTorch 2.10.0+cu128 with working `sm_120`
kernels. Qwen3-1.7B GPU inference and two real language-model LoRA publications
were executed. The fixture-operator run succeeded on 5/8 tasks, while the all-model
operator run admitted no memories because its Writer candidates failed evidence
checks. Full measurements and their limitations are in [local model notes](local_model.md).

A grouped exploratory stream subsequently completed 182 of 500 planned tasks
before its local Writer allowance stopped it. It recorded 38 adapter publications,
zero HTTP/background errors and a 9.61 GiB CUDA peak. The largest post-maintenance bank sample was only 46
tokens, so capacity maintenance was not tested. End-state comparisons and the
training/protocol limitations are documented in the [pilot report](exploratory_chain_2026-10-23.md).

The Groq GPT-OSS 120B Writer was subsequently connected to the local task model
with fixture consolidation: all four tasks completed, four Writer memories were
admitted, and one real LoRA publication completed without background errors.
The run used 4,809 Writer tokens and peaked at 9.09 GiB CUDA allocation. This
verifies the mixed backend path, not a measured learning advantage.

## Not yet validated

- Formal remote-model loading, chat templates, LoRA target modules and vLLM
  adapter hot loading. Local Qwen3-1.7B has been checked separately; Qwen3.5-9B
  was an earlier candidate, while formal model selection remains open.
- Hugging Face SFT against the actual remote target models. The complete backend
  has been exercised on local Qwen3-1.7B, not on the formal remote models.
- Native interface checks do not complete fixed benchmark manifests or the domain
  verifier. Automatic semantic certification remains open and is gated.
- An actual trained Auto-Dreamer Consolidator checkpoint and faithful reproduction.
- Long-run resource/service budgets, inference contention, remote crash recovery,
  quantitative advantage over baselines, and any contribution claim.

The model, task and validation boundaries are explicit so these checks can be
completed without rewriting memory management. Each integration must update this
document and the relevant directory README.

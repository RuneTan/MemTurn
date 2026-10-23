# Grouped exploratory task stream

This is a qualitative development check, not a formal continual-learning result.

The current runner freezes a protocol version as well as settings. The original
2026-10-23 stream used the earlier prompt/training protocol; it cannot be resumed
under the repaired code as if the method were unchanged. Preserve that run and
start a new output directory for the repaired protocol. Reproducing the original
resume requires its original code revision.
The plan contains 500 tasks, grouped in blocks of 20. Conditions cycle through
`vault:v1`, `api:v1`, `vault:v2`, and `api:v2`, then recur. These are four known
rules within two synthetic families, not 500 distinct skills or benchmark tasks.

The local task model is Qwen3-1.7B; Groq GPT-OSS 120B is the Writer. Consolidation
uses the deterministic development implementation. Fixed-rank LoRA continues
from the published version, with the existing two-step SFT job every eligible
four-task batch. All task and memory budgets match the local profile.

## Quota policy

The preceding pilot used about 1,200 Writer tokens per task. A 200k daily allowance
would support about 160 such tasks before earlier account usage and variation.
The actual run caps its reported uncached Writer tokens at 180k, reserves a small
margin before another request, and respects actual provider quota errors. Cached
tokens are deducted only when the provider explicitly reports them. The provider's
accounting remains authoritative.

The run stops at its configured budget or provider quota, saves the completed-task
count and any deferred Writer source, and does not upgrade billing or substitute
fake experiences. Minute limits get bounded retries. The full planned manifest
and the number actually executed are reported separately.

The token allowance is per explicit launch. A later `--resume`, requested after
quota restoration, starts a new local allowance while retaining cumulative usage,
task count, adapter weights and memory. It does not reset the provider's quota.

## Observations

- Every task: reward, completion, first action, steps, memory occupancy, evidence
  storage, adapter count, CUDA allocation and sampled process RSS.
- Every Writer call: original final response and returned usage in the audit.
- Every decoded candidate: full pre-validation memory record, including rejected
  candidates. Completed source trajectories remain available in the audit.
- At startup and each group boundary: paired first-action probes on fixed held-out
  objects, under base/current adapter and absent/current memory. Probes provide no
  feedback, create no experience and do not refresh memory recency.

The same-bank probe comparisons are diagnostics, not four independent learning
baselines. Their direct-first-action metric counts a search request as not yet a
correct environment action. The training observations may contain explicit
corrective feedback, which is why no-feedback probes are reported separately.

## Run and resume

With `MEMTURN_WRITER_API_KEY` configured in the launching process:

```text
.venv-local/Scripts/python.exe run_memturn.py long-chain --output runs/your_chain --tasks 500 --group-size 20 --writer-token-budget 180000
```

After a clean pause, use the same command with `--resume`. The configuration
fingerprint must match. A deferred Writer source is retried without counting or
executing another task. This implements clean-boundary recovery, not arbitrary
crash-exactly-once service; an inconsistent database/checkpoint pair is rejected.

`checkpoint.json` and `report.json` are replaced atomically after maintenance.
`metrics.jsonl` records per-task measurements; `probes.jsonl` records probes.
`bank_*.json` and the audit are observer artifacts outside agent-accessible state.
They are included in experiment-output storage, not silently offered as memory.

Offline analysis, with Matplotlib installed in the chosen analysis environment:

```text
python scripts/summarize_chain.py runs/your_chain
```

This creates `analysis/summary.json`, a Chinese group report, Writer examples with
evidence, and PNG/SVG charts. It can also inspect progress while the run is active.
No model calls are made during analysis.

After the chain pauses, a separate local-only complete-task diagnostic can run:

```text
.venv-local/Scripts/python.exe scripts/evaluate_chain.py runs/your_chain
python scripts/summarize_chain.py runs/your_chain
```

This executes 32 held-out episodes across the same four model/memory settings,
checks that learning state is unchanged, and records full action traces. It makes
no Writer requests and trains nothing. Group-end memory may be empty after
internalization; overlapping memory/no-memory results then cannot identify a
memory effect.

An additional `--show-tool-state` option gives the policy the remaining search
count, previous query and returned references. It supplies no correct action.
Its separate result files compare this visibility intervention on the same
end-state tasks. The original main stream is left unchanged; this diagnostic
does not establish a production tool-history implementation.

# Grouped-chain pilot: 2026-10-23

[English](exploratory_chain_2026-10-23.md) | [简体中文](exploratory_chain_2026-10-23.zh-CN.md)

## Scope and stopping point

The plan contained 500 tasks. The run stopped cleanly at **182 tasks** near its
180k Writer-token allowance, with a resumable checkpoint. This was a local budget
stop, not an observed provider daily-limit rejection. No paid-plan upgrade occurred.

Twenty-task groups cycled through four known rules in two synthetic families:
`vault:v1`, `api:v1`, `vault:v2`, `api:v2`. Qwen3-1.7B executed tasks, Groq GPT-OSS
120B wrote memories, and the deterministic fixture Consolidator generated training
material. Maintenance completed between tasks. These are development diagnostics,
not ScienceWorld/ALFWorld results, capacity-pressure evidence or an online-service test.

| Measurement | Observed |
| --- | --- |
| Completed episodes | 182; 153 succeeded (84.1%) |
| Mean reward | 0.661; cumulative 120.3 |
| Writer calls | 182 successful; zero HTTP/background errors |
| Memories admitted / null responses | 153 / 29 |
| Consolidations / adapter publications | 45 / 38 |
| Writer input / output / reported cached tokens | 165,783 / 47,767 / 37,376 |
| Writer budget estimate after reported cache deduction | 176,174 tokens |
| Main-chain elapsed time | 37.6 minutes, including throttling and probes |
| Peak CUDA allocated / sampled post-maintenance RSS | 9.61 / 2.19 GiB |
| Largest post-maintenance active-memory sample / cap | 46 / 8,000 tokenizer tokens |
| Maximum adapter weight files after maintenance | 1 |

At the observed rate, 500 similar tasks would need roughly 484k uncached tokens.
This is an estimate for these small trajectories, not a quota guarantee for other
tasks. The run may be resumed explicitly after an allowance is restored.

## Repeated task groups

| Condition | First 20-task visit | Second 20-task visit | Third visit |
| --- | --- | --- | --- |
| vault:v1 | 20/20 | 20/20 | 20/20 |
| api:v1 | 16/20 | 20/20 | 2/2, incomplete group |
| vault:v2 | 14/20 | 20/20 | — |
| api:v2 | 1/20 | 20/20 | — |

Practice coincided with better episode completion. This stream alone cannot
attribute the change to LoRA, external memory, corrective feedback or their
interaction. In particular, API v2's second visit completed every task even though
its first environment action was never correct.

## Findings and limits

**The Writer was operationally reliable here.** All 153 nonempty candidates
passed source checks; the 29 null outputs matched trajectories with no accepted
preparation action. Their admitted content consisted only of the relevant command
name plus a condition label. This does not test rich summaries or broad semantic
correctness. Raw responses, candidates and source trajectories are retained in
observer logs, outside retrievable agent state.

**An unchanged average hid an action-prior replacement.** No-memory first-action
probes stayed at 25% overall, but the final adapter began with `inspect_lock` for
all four conditions; the base began with `check_permission`. Vault v1 accuracy
rose while vault v2 fell on these fixed probes. The parameters changed behavior,
but conditional mastery of all four rules was not demonstrated.

**The tiny SFT schedule selected a biased slice of its data.** In the first group,
20 examples were available across five batches. Two optimizer steps with one
example per step consumed only the first two chronological examples per batch:
10 examples total, eight already containing explicit corrective feedback. Thus
update/publication counts must not be interpreted as verified internalization.
This is a reconstructed audit of the first group, not an assumption about every
later mixed bank. Sample coverage and no-feedback behavior should be checked
before retiring memory.

**Retrieval loops were a concrete execution problem.** All 29 failed episodes
contained `invalid_memory_search_budget` actions. Search calls/results and the
remaining budget were not explicitly represented in the policy history. A later
diagnostic added state hints but did not improve the base-with-memory completion
rate. Missing state is therefore an interface concern, not a proven sufficient
explanation or a completed fix.

**Memory can interfere well before capacity is reached.** End-state probes used
the same eight held-out tasks and the same current bank:

| Policy / memory | Success | Mean reward |
| --- | --- | --- |
| Frozen base, no memory | 8/8 | 0.900 |
| Final adapter, no memory | 8/8 | 0.875 |
| Frozen base, current memory | 4/8 | 0.175 |
| Final adapter, current memory | 8/8 | 0.875 |

These paired checks suggest improved end-state behavior under this memory context,
not a general advantage from learning. A second held-out object-ID range gave
6/8 versus 8/8 for base/current adapter with memory, while no-memory completion
remained 8/8 for both. The current lexical retriever returns unrelated entries;
small-model behavior and the tool protocol remain important confounders.

**Capacity maintenance was not exercised.** The largest post-maintenance memory
sample was 46 tokens, far below 8,000; this is not an intra-task peak measurement.
Four recurring rules and frequent consolidation/internalization keep the bank
small. This cannot validate the proposed capacity mechanism or delayed diversity
collapse. More distinct, persistent knowledge is needed for that question.

## Next priorities

1. Represent real retrieval tool calls/results and availability in the policy
   history; test wrong-condition retrieval explicitly.
2. Cover the chosen experience's training samples and reduce chronological/feedback
   bias; log the examples actually consumed by each optimizer job.
3. Separate successful parameter updates from evidence of no-memory use before
   external memory exits.
4. Move to richer tasks and capacity-pressure tests after the interface and training
   checks work. Add independently evolving matched-budget baselines for formal claims.

## Artifacts

Local run: `runs/chain500_group20_20261023/`. Its checkpoint, metric/probe JSONL,
raw audit, bank snapshots, charts and end-state traces remain ignored experiment
outputs. `analysis/initial_training_audit.json` records the first-group sample
reconstruction. See [runner and analysis instructions](grouped_chain.md).

The implementation regression suite passed **38 tests**. End-state diagnostics
additionally checked that their evaluation did not alter the learning state.

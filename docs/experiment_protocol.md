# Continual-learning experiment plan

[English](experiment_protocol.md) | [简体中文](experiment_protocol.zh-CN.md)

Updated: 2026-10-29. **This is the planned evaluation, not a report of completed experiments.** The prototype implements the memory and training lifecycle; several interfaces below still need implementation and acceptance.

## Research question

Can an agent turn its own experience into useful long-term capability while keeping external memory, trainable parameters and learning costs bounded? The intended comparison measures three things together: useful work delivered, retention of still-relevant skills, and the ability to learn new tasks after a long history.

The default remains a frozen base model, one fixed-rank LoRA adapter, grounded experience writing, cross-task consolidation and selective internalization. Each regular batch of four completed source task instances selects at most one consolidated entry supported by at least two distinct instances. Training uses one rule QA plus verified source decisions, with one complete pass through eligible samples. Successful adapter publication retires the selected entry; it is not a test that the experience has been mastered. Cold-region capacity maintenance is separate and does not train.

## Environments and readiness

| Environment | Evaluation unit | Current status |
| --- | --- | --- |
| ScienceWorld | One episode | Native wrapper and narrow observed-transition verification exist; formal streams and full acceptance remain pending |
| Continual Learning Bench | One database question or code issue | Planned; needs a worker, action bridge, domain verifier and versioned task streams |
| MemoryArena | One complete multi-session chain | Planned; needs a worker, parent-chain coordinator, domain verifier and chain-level scoring |
| ALFWorld | One episode | Existing integration remains available; it is outside the current three-environment primary plan |

Start with ScienceWorld, then add [Continual Learning Bench](https://github.com/pgasawa/continual-learning-bench) and [MemoryArena](https://github.com/ZexueHe/MemoryArena). Preserve each benchmark's native scores alongside the common metrics. New changing streams and development/test splits must be identified as extensions, with dataset revisions and manifests recorded.

For MemoryArena, each child task has a separate source ID, but utility is settled once per parent chain. Regular consolidation still counts source instances, not complete chains. Report distinct-chain support as well as distinct-task support; two subtasks are not two independent chains. Keep chain order, a shared parent deadline and the complete scoring denominator, including unattempted subtasks. Report the native PS/sPS and success rate.

For Continual Learning Bench, start with database exploration. Preserve the correct database snapshot or code revision; an error against an obsolete schema is not by itself evidence of forgetting. Small native task pools require explicit extensions before supporting long independent streams. Hidden answers, reference SQL and patches must not leak into public experience or training inputs.

## What the evaluation measures

### Useful work under a fixed service horizon

Freeze the service duration $H$, all arrivals $a_i$, deadlines $d_i=a_i+D_i$, and the native quality mapping $q_i\in[0,1]$ before comparing methods. Let $t_i$ be the actual valid terminal-return time:

$$
z_i=\mathbf1\{\text{valid terminal return and }t_i\le d_i\le H\},\quad
u_i=z_iq_i,\quad U_H=\sum_{i\in\mathcal I_H}u_i.
$$

Rejected, timed-out and failed requests remain in the offered-request denominator with zero utility. Valid, timely partial progress uses the frozen native quality mapping. Stop new arrivals after $H-D_{\max}$; the common tail stays inside $H$. At $H$, stop accepting results and new learning work, cancel or freeze unfinished work, and prevent later adapter publication. Record all consumption, including cleanup. One append-only analysis ledger records every parent request; learning cannot retrieve it.

Also report $U(t)=\sum_i u_i\mathbf1\{t_i\le t\}$ against elapsed service time. All serving-time writing, consolidation, training, loading and checks consume the common budget; initial service startup precedes the clock.

### Learning over the offered stream

Using the final settled utilities in arrival order, for the same $N$ offered requests:

$$
C_n=\frac1n\sum_{i=1}^n u_i,\qquad
\mathrm{AUC}_{\mathrm{stream}}=\frac1N\sum_{n=1}^N C_n.
$$

Report $U_H$, the $C_n$ curve and this area together. The area emphasizes earlier utility; it is not an integral of $U(t)$ or the adaptation area below. For $u=[1,0.5,0,0]$, $U_H=1.5$ and $\mathrm{AUC}_{\mathrm{stream}}=0.65625$.

### Retention of valid old skills

For each task cohort $j$, fix an independent test set and a common reference checkpoint $a_j$. At checkpoint $k$, include only older cohorts whose test conditions remain valid:

$$
\mathcal O_k=\{j:a_j<k,\ j\text{ remains valid at }k\},\qquad
\mathrm{BWT}_{\mathrm{valid}}(k)=\frac1{|\mathcal O_k|}\sum_{j\in\mathcal O_k}[R_j(k)-R_j(a_j)].
$$

$R$ is mean native quality. Report absolute scores and per-cohort changes as well. An empty set is NA. Keep a fixed early cohort in the main retention plot so changes in cohort composition cannot hide forgetting. Test returning tasks before new feedback. Old conditions that remain in scope need their original environment version; retired conditions follow a shared validity manifest.

### Ability to learn after prolonged operation

Reserve challenges $\mathcal Q$ that never enter the main stream. Each has separate adaptation and test instances. At different stream ages $k$, start independent learning branches from the published model, committed memory and necessary sources. Give each branch the same fresh budget, challenge set and feedback schedule, without pending work from the main stream.

Measure each challenge before learning and after fixed counts $b$ of terminal feedback, including success and failure:

$$
P_j^{[k]}(B)=\frac1B\int_0^B[R_j^{[k]}(b)-R_j^{[k]}(0)]\,db,\qquad
P_{\mathrm{age}}(k;B)=\frac1{|\mathcal Q|}\sum_{j\in\mathcal Q}P_j^{[k]}(B).
$$

Capture the published state at each feedback checkpoint without waiting for background work to drain. Use trapezoidal integration over these frozen checkpoints. Report starting score, final score and elapsed adaptation time alongside the gain area: a high starting score leaves less room to improve. If any challenge fails to reach $B$ within the common time limit, mark the aggregate NA and report coverage. Do not shorten $B$ or drop unfinished challenges.

This measures the learning ability of the whole memory-plus-model system. It does not by itself establish parametric internalization. Tests keep parameters fixed; multi-session tests may use disposable within-chain memory. No probe data or updates return to the main stream or adaptation branch.

## Checkpoints and causal comparisons

Checkpoint positions are fixed by offered parent-request counts. Immediately before dispatching request $m_k+1$, atomically capture the currently published state; take the final checkpoint at $H$. Do not wait for background work to drain. Report completed requests, feedback, publications and time at each point. Snapshot export costs count in the main run; subsequent isolated probes have a separate research-compute budget.

Evaluation cohorts are separate from the code's environment-valued `Task.family`; record `cohort_id`, `variant_id` and `reference_checkpoint` explicitly.

Start with four implemented configuration arms:

| Arm | Comparison |
| --- | --- |
| `raw_external_only` | Verified external memories with simple cold eviction; no consolidation or SFT |
| `consolidation_only` | External consolidation without parameter learning |
| `raw_experience_sft` | Verified raw decision SFT; one-source eligibility |
| `baseline` | Full MemTurn with mixed material, support-based selection and retirement after publication |

After matrix generation, explicitly set `raw_experience_sft` to `capacity_policy=evict` to match the raw external-memory arm. This is cold-region eviction, not FIFO. Add `train_without_exit` first to test retirement, then selected material/selection/capacity ablations. The existing matrix tools generate and execute profiles; they do not yet implement this full evaluation protocol.

Use the same base and role models, actions, arrivals and resource ceilings. Report actual consumption; equal ceilings do not imply equal training. For internalization, separately hold the source pool and training target fixed, and compare before/after training with/without the target memory on independent new instances. Hide its parents, same-source entries and identified copies as well. Keep this fixed-experience diagnostic distinct from long-stream system comparisons.

## Deployment plan and remaining work

The planned allocation is eight H20 GPUs in two independent four-GPU slots. Each slot has task inference, Writer, Consolidator and single-GPU SFT, and runs one arm/seed at a time. Keep directories, ports and quotas separate. Unused training capacity is not reassigned to inference for other arms. This is not eight-GPU distributed training: the current trainer has no DDP support.

Qwen3-8B BF16 for the task model and Qwen3-32B for each fixed operator are development candidates, not selected or validated formal models. Confirm VRAM, interconnect, scheduler, runtime limits, GPU-hour allowance, model revisions, prompts, inference settings and single-GPU fit before freezing the deployment. The initial plan uses five paired independent streams; final lengths and budgets follow development calibration.

Before formal experiments, implement and accept:

1. A horizon-aware arrival driver, complete parent-request ledger, terminal timestamps and cutoff handling. Current periodic-arrival and timeout support does not provide these guarantees.
2. Continual Learning Bench and MemoryArena workers, action bridges, native scoring and evidence checks, including parent-chain coordination.
3. Consistent checkpoint export, fresh learning-branch construction and isolated retention/adaptation probes. Copying a live SQLite file or resuming it with a different stream is insufficient.
4. The metric outputs above and per-environment acceptance: hand-checkable utility/AUC, unchanged main-state hashes after probes, faithful native scores, traceable training material, three continued publications, capacity pressure, timeout and recovery.

See [engineering handoff](cluster_handoff.md), [validation status](validation.md) and [literature](literature.md). No benchmark advantage, sustained plasticity or novelty claim has yet been established.

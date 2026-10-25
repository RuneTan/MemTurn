# Implementation status: 2026-10-24

[English](implementation_2026-10-24.md) | [简体中文](implementation_2026-10-24.zh-CN.md)

The agreed mixed-supervision candidate is implemented. Optimizer execution is
not evidence that the selected experience has been internalized.

**Audit correction:** The 62-test and two-job results below are historical.
The old QA probe mistakenly requested environment-action JSON; its output cannot
diagnose QA failure. See the [engineering audit](audit_repairs_2026-10-24.md)
for the corrected probe, new regression results and three continued publications.

## Training material

The regular Consolidator emits content, parents, evidence refs, a rule question
and source-linked decisions in one operation. The current prompted operator is
not Auto-Dreamer's trained checkpoint. No additional teacher or rewriting agent
has been added.

Each `Example` records `example_type`, `output_kind`, `memory_ref`,
`covered_content_refs` and `support_refs`. The decoder binds relative `content`
or `span:start:end` refs to the emitted memory version. Decisions use `task_id`
and zero-based `step` refs and copy original observations/actions verbatim.
No default state projection or environment-specific answer masking remains.

Rule QA receives a question without the selected memory body; its text target is
copied from the covered content. Decisions predict environment-action JSON using
the original public observation and preceding environment history. Historical
retrieval bodies/queries are excluded from SFT inputs. Other allowed memory is
pre-retrieved after excluding the selected version, parents and same-source
aliases. The static forward/backward pass makes no new tool calls.

Default `training_material: mixed` requires both types. The existing
`max_examples_per_job: 32` budget includes QA. Questions add no independent task
support. Invalid training material is stripped while otherwise valid external
memory can remain. Invalid memory evidence itself prevents admission/replacement.
No silent fallback to decision-only training occurs. The tiny classifier and
historical action-only diagnostics explicitly use `decision_only`.

Selection still chooses at most one eligible memory by verified distinct-task
support. One shuffled epoch with accumulation one continues fixed-rank LoRA.
After candidate preparation, publication directly retires the selected version.
The agreed baseline has no behavioral mastery gate; this change adds none.

## Retrieval and growth

Episodes persist automatic and additional retrievals separately from environment
steps. The Writer sees the memory actually exposed to the task policy. Native
private manifests and dataset condition labels are not sent to operators.
Bounded retrieval history remains in task context; duplicate bodies appear once.

`stats.growth` records bounded cumulative token counters:

```text
active_tokens = initial_tokens + admitted_tokens
                - regular_released_tokens - internalized_tokens
                - capacity_released_tokens
```

Release means before-minus-after and can be negative for a growing regular
replacement. Retries do not double-count and transactions roll back counters
along with memory changes. Existing databases start from an initial baseline.
This measures growth; it does not activate growth-based adaptive consolidation.
Fixed-cycle regular maintenance and cold-region/half-budget capacity maintenance
remain in place. Capacity events do not train.

## Original environments

Windows model dependencies and Linux simulator dependencies communicate through
`ProcessEnvironment` / `native_worker.py`, a JSON-line process boundary. It
transmits original observations, actions, feedback and rewards; no learning or
memory policy is embedded in the worker. No global Python/Java install changes.

`install_native_linux.py` creates `.venv-native-linux` with ScienceWorld 1.2.3,
ALFWorld 0.4.2 and PyYAML 6.0.3, saves resolved dependencies, and obtains portable
Java 17 from the [official Adoptium API](https://api.adoptium.net/). The Java
archive is SHA-256 checked; its resolved URL/checksum are saved in `.cache/native`.
`prepare_native_checks.py` uses the official ALFWorld text assets, records their
origins/checksums and creates explicit training-split interface manifests.
Expert plans are not exposed to the model. ScienceWorld uses no simplifications.
Dependencies, assets, raw runs and weights stay out of Git.

Inside Linux/WSL, from the checkout:

```text
python3 scripts/install_native_linux.py
.venv-native-linux/bin/python scripts/prepare_native_checks.py --kind scienceworld --wsl-distro YOUR_DISTRO
.venv-native-linux/bin/python scripts/prepare_native_checks.py --kind alfworld --wsl-distro YOUR_DISTRO
```

Generated process commands target the named WSL distro. On the Windows host:

```text
.venv-local/Scripts/python.exe scripts/check_native.py --config runs/native_setup_20261024/scienceworld.check.json --output runs/NEW_scienceworld_check
.venv-local/Scripts/python.exe scripts/check_native.py --config runs/native_setup_20261024/alfworld.check.json --output runs/NEW_alfworld_check
```

For native Linux, replace `environment.worker_command` with the absolute Linux
paths to the isolated Python and `scripts/native_worker.py`. Interface checks
call no models and perform no training. These manifests are not benchmark runs.

`NativeReplayVerifier` replays source observations, feedback, evidence, reward
and success. An admissible action is not necessarily a good training target.
Replay does not certify arbitrary natural-language generalizations: an
independent domain predicate is required for memory support, otherwise support
fails closed. `verification_factory` already supports that replaceable boundary.

## Evidence and remaining work

Both ScienceWorld and ALFWorld completed two original training-instance
reset/step checks each, without model calls or training. Records are in
`runs/native_scienceworld_interface_20261024/` and
`runs/native_alfworld_interface_20261024/`; they are not task-solving results. Two ScienceWorld records and one ALFWorld
record also matched independent native replay; summary support without a domain
predicate correctly remained false. ALFWorld interface runs took about 55–105
seconds each on this WSL-mounted checkout, largely initialization; these are not
online service measurements.

Regression: **62 tests pass**, including mixed-target ownership/fidelity, no
silent fallback, public/private separation, replay tampering, process restart,
growth balance and rollback.

Qwen3-1.7B on the RTX 5080 Laptop completed two continuing rank-8 LoRA jobs.
Each used one diagnostic QA and two unchanged ScienceWorld decision records:
three optimizer updates, full sample coverage, frozen base weights. Adapter
preparation and continued parent-version training succeeded. Raw output:
`runs/mixed_sft_interface_20261024/report.json`.

The QA is an explicit interface fixture, not a domain-verified Consolidator
conclusion. No external memory was retired. The post-training QA probe still
returned an environment-action object. Mixed targets enter training, but reliable
rule-text behavior and native end-to-end internalization remain unproven. This
does not establish learning benefits or appropriate hyperparameters.

Still required: domain evidence verification, real operator quality checks,
task-solving runs and long-chain matched baselines; remote hardware and
Qwen3.5-9B serving/LoRA compatibility; measured online service contention.
No teacher, adaptive controller, RL or mastery gate was silently introduced.

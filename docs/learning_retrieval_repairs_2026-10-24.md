# Learning and retrieval investigation: 2026-10-24

[English](learning_retrieval_repairs_2026-10-24.md) | [简体中文](learning_retrieval_repairs_2026-10-24.zh-CN.md)

Historical repair snapshot (source-state projection is retired; see [current implementation](implementation_2026-10-24.md)): the later 2026-10-24 decision replaced the fixed-step
default with one full sample pass and accumulation one. Its current behavior is
documented in [configuration](../configs/README.md). The measurements below are
preserved; automatic accumulation growth is no longer the current default.

Retrieval had reproducible interface defects; training had sample-coverage and
objective problems. Those implementation defects have been repaired and targeted
local execution has improved. Effective parametric learning remains incomplete.
These are development checks, not formal paper evidence.

## Evidence

The original `runs/chain500_group20_20261023/` remains intact. The investigation
checked code, metrics, raw responses, trajectories and adapter publications.

| Issue | Observation | Interpretation |
| --- | --- | --- |
| Partial chronological sampling | First group offered 20 examples but consumed 10; all 38 updates used two steps | Step and microbatch limits left data uncovered |
| Answer-bearing inputs | Eight of those ten consumed inputs contained `Required next action` | Valid correction supervision does not establish clean-start recall |
| Incompatible retrieval | 158 tasks received memories; 64 received another condition's entries | Lexical overlap did not exclude incompatible toy versions |
| Invisible searches | Search calls/results were omitted from policy history | Repeated requests were not clearly grounded in a transcript |
| Tool errors sent to simulator | 55 tasks contained `invalid_memory_search_budget`, including all 29 failures | Tool and environment feedback were conflated |
| Copied placeholders | Three failures in an initial eight-task repaired run repeated `one environment command` | The model copied a schema example as an action |

These counts are exposure statistics, not causal effects. Some tasks with tool
errors recovered. Source validation, assistant-only loss masks, frozen base
weights, fixed-rank adapters and real gradient updates were checked; the evidence
does not support a claim that training never occurred.

## Changes

**Training coverage.** Shuffle examples, cover every selected example before
repeating, and increase accumulation when required to fit the configured optimizer
steps. Token and deadline limits remain binding. Incomplete coverage rejects the
job before publication and memory exit. Artifacts record consumed source/input
steps, token counts and input hashes. This covers selected examples, not every
support trajectory.

**Verified input projection.** `Example.step` retains the accepted evidence step;
optional `input_step` identifies the training input. Only the toy adapter projects
an accepted action to its original initial observation, after simulator replay
verifies that action there. Training history stops at the input step. Source
trajectories are not rewritten and no teacher is added. Other domains need their
own independent replay/domain verifier.

**Visible tool protocol.** Automatic retrieval, subsequent queries/results,
remaining calls and errors have a separate policy transcript. Environment evidence
indices remain stable. Duplicate queries can reuse results; calls remain bounded.
Exhausted requests return to the policy and are never fabricated simulator actions.
Persistent invalid tool requests can still fail at the decision limit.

**Optional applicability scope.** The toy adapter derives incompatible family and
version scopes from public observations, without using the correct-action table
to choose results. Unscoped domains retain lexical retrieval. This is not a
universal exact-match restriction on transferable knowledge or a solved semantic
retrieval method.

**Prompts and evaluation.** Describe action/query fields without copyable command
placeholders. Memory-disabled evaluation removes both automatic and additional
retrieval, including tool-history leakage. Training and inference share formatting;
full training conversations omit the generation prompt, consistent with the
[official Transformers guide](https://github.com/huggingface/transformers/blob/main/docs/source/en/chat_templating.md).

## Local checks

The final-format ablation fixed the base model, original final bank and eight objects
unused by gradient training:

| Setting | Completed | Mean reward |
| --- | --- | --- |
| Original implementation | 4/8 | 0.175 |
| Scope only | 6/8 | 0.550 |
| Full tool protocol only | 8/8 | 0.925 |
| Both | 8/8 | 0.950 |

The protocol is a bundle: call/result history, automatic-retrieval explanation,
dynamic availability and error routing. The comparison does not identify one
prompt sentence's independent effect. An earlier status-hint-only intervention
had remained at 4/8.

Removing placeholders then improved a separate eight-task complete runtime check
from 5/8 to 8/8, and mean reward from 0.325 to 0.950. It used real task inference
and LoRA with fixture Writer/Consolidator; adapter publications changed from one
to two and no background errors occurred. Subsequent trajectories and training
also changed, so this is not an isolated long-term learning effect.

The final-format retrieval repeat is stored in `runs/repair_protocol_v3_20261024/`,
with per-setting results in `validation_summary.json`.

The first four projected examples all reached real optimizer updates without
corrective feedback in their inputs. A separate offline calibration collected
two examples per condition from existing trajectories and held the eight-example
dataset, pre-placeholder-removal template and initial seed fixed. Two steps achieved 2/8 correct first
actions on the fixed diagnostic objects; 32 steps achieved 4/8. Both covered all
examples, processing 1,476 and 5,904 tokens respectively. A separate two-step check
with the final template also achieved 2/8. This does not establish
an optimal schedule or justify changing the default automatically. It is an
offline calibration, not a comparison of online learning systems.

## Remaining limits

1. Coverage and weight updates do not establish mastery. Publication still lacks
   a general behavioral mastery gate; safe deletion has not been demonstrated.
2. The 32-step diagnostic reaches only 50%. Data diversity, optimization, conditional
   binding and model capability remain possible explanations.
3. Fixed-bank checks do not establish evolving-memory long-run behavior. These
   objects were reused during debugging, so they are not a final independent test
   set. Real benchmarks, capacity pressure and uninterrupted serving remain open.
4. The controller bounds tool misuse but never substitutes the correct action for
   the model.

## Reproduction and artifacts

```text
.venv-local/Scripts/python.exe scripts/validate_repairs.py --output runs/new_repair_check
.venv-local/Scripts/python.exe -X utf8 -m unittest discover -s tests -t . -v
```

The script uses only local models, never Groq, and does not resume the source run.
It loads comparison code from commit `c0a1d01` and saves source hashes, legacy
implementation, traces, examples and training metrics. The current long-chain
protocol is `retrieval-transcript-v3-sft-coverage-v2`; old checkpoints cannot be
silently mixed with it. Use a new output directory for new comparisons.

- `runs/repair_validation_20261023/`: investigation and first ablation.
- `runs/repair_balanced_two_step_20261024/` and `runs/repair_balanced_calibration_20261024/`: fixed-data calibration.
- `runs/repair_runtime_20261024/` and `runs/repair_runtime_schema_v3_20261024/`: complete-runtime placeholder comparison.
- `runs/repair_protocol_v3_20261024/`: final protocol repeat and real SFT check.

Code: [training](../src/memturn/learning/huggingface.py),
[prompts](../src/memturn/models/http.py), [runtime](../src/memturn/runtime/service.py),
[retrieval](../src/memturn/memory/retrieval.py), [toy verification](../src/memturn/tasks/toy.py),
[validation script](../scripts/validate_repairs.py).

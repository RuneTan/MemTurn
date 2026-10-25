# Engineering audit and repairs (2026-10-24)

[English](audit_repairs_2026-10-24.md) | [简体中文](audit_repairs_2026-10-24.zh-CN.md)

> This document preserves the initial audit and archive-recovery record. Test counts and incomplete items describe those stages. Later recovery, timeout, budget, domain-check and comparison tools are covered by the [current cluster handoff](cluster_handoff.md) and [validation record](validation.md).

The audit baseline was `d631b22`. The repairs and local checks below do not establish benchmark capability, long-run benefit or service latency.

## Repairs

- Retrieval calls no longer consume environment action steps. Exhausted retrieval returns an empty result with a bounded opportunity to correct the request.
- Real retrieval changes the access sequence even within a batch, allowing capacity commits to reject stale cold rankings. Oversized cold entries no longer block later usable entries.
- Regular jobs record the batch at their start. Writer failures still advance maintenance, and failed consolidation of a new region does not block selection of existing eligible candidates.
- Task results and pending sources are registered atomically; external replacement and its commit record are atomic; the model pointer, selected-memory exit and publication record are atomic. The current publication record survives historical-job rotation, and recovery never treats READY as published.
- Writer supports must come from the current task. Verifier errors reject certification, and mixed candidates require exactly one rule question. The coordinator checks output and example budgets independently of the operator.
- Training retrieval excludes identical content from independent sources. Direct restatement in inputs or action history rejects training while retaining memory; source states are not rewritten. Indirect leakage through semantic paraphrases still requires domain checks.
- A retired model still referenced by an old task blocks the next training job. Adapter unloading no longer holds the foreground database lock. Repeated close is safe, and shutdown waits for accepted tasks to enqueue their sources.
- Evaluation-only Runtime does not run recovery, Writer or maintenance. Releasing an evaluation snapshot does not garbage-collect learning sources.
- Queued payloads count toward the source budget. Consolidation reserves workspace; HTTP requests and responses have byte limits; large Writer requests reserve capacity based on their actual size.
- LoRA continuation/loading checks the base revision, rank, alpha, target modules and file hashes. Post-update numeric checks, gradient statistics, target-token counts and reproducible initialization were added.
- Unregistered model directories after restart count toward the version limit. At this audit stage, reaching the limit prevented new training and preserved checkpoints for review rather than deleting them automatically.

## Correcting the question-answer diagnostic

`check_mixed_sft.py` previously called `training_messages(qa.prompt)` for a rule question. Its default requested action JSON, so an action-shaped answer could not establish failure to learn question answering.

The call now explicitly passes `qa.output_kind`. The corrected two-job check still used one question, two native-source actions and three updates per job, with 100% sample coverage. Under the correct text prompt, the model returned prose. The answer was general and reached the 96-token output limit; rule mastery was not established and no external memory exited. Original historical artifacts were retained.

Record: `runs/audit_mixed_sft_20261024/report.json`.

## Validation results

At this stage, **93 tests passed** in `.venv-local`, including 31 added failure and concurrency checks. A Python environment without optional training dependencies skips the corresponding two tests; the project environment provides the complete result.

A separate check used Qwen3-1.7B, real rank-8 LoRA and deterministic experience operators on 12 toy tasks:

| Item | Result |
| --- | --- |
| Tasks completed | 12/12; cumulative development-environment utility 11.3 |
| Regular consolidations and model publications | Three each, no background errors |
| Samples / updates in the three jobs | 3/3, 5/5, 3/3; full coverage |
| Parameter continuation | base → first LoRA → second → third |
| Final active memory | Two entries, 28 tokens |
| Active-text accounting | 1908 − 1838 − 42 − 0 = 28 |
| Final model directories | One current version |

Records: `runs/audit_local_runtime_20261024/report.json` and `audit.jsonl`. This entry point waits for background work between tasks, so it is not an online load test. Experience operators are development fixtures, not a quality test of language-model Writer/Consolidator output. The real-model run did not trigger capacity maintenance; failure regressions cover that path.

Recheck with:

```powershell
.\.venv-local\Scripts\python.exe -X utf8 -m unittest discover -s tests -t . -v
```

If the host temporary directory is not writable, point that process's `TEMP` and `TMP` at a newly created temporary directory inside the project.

## Boundaries not yet accepted at that stage

Native free-text domain verification, real experience-operator quality, matched-budget long streams, arrival load and tail latency, forcibly cancellable background jobs, and joint hard GPU/RAM/disk budgets had not passed complete acceptance. SQLite page limits applied only to the main database, with WAL/SHM reported separately; this was not a total-disk bound. Remote tokenizer and deployed-model identity records still needed to be frozen.

Task deduplication then relied on a bounded recent-result window. Arbitrary old IDs outside it did not have permanent exactly-once protection. A formal recovery protocol needed stream position/high-watermark state and a replay window; research logs could not act as an unlimited runtime deduplication or source database.

Subsequent [orphan archive and restore](adapter_archive_recovery.md) support copies and fully verifies candidates before cleanup, while protecting current and in-flight versions. Unknown directory names remain. Eleven added checks brought that stage to **104 passing tests**. Historical runs with databases contained no orphan candidates, so no bulk cleanup of old artifacts was performed.

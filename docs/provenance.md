# Lightweight provenance checks

[English](provenance.md) | [简体中文](provenance.zh-CN.md)

Updated: 2026-11-01. `Validator(store)` checks source provenance and training-data fidelity. It does not certify that a source entails a summary. No LLM judge, full-episode replay, literal summary matching or positive-score requirement is used.

## Citation lifecycle

The application records each episode with a unique task ID and zero-based steps. Writer proposes a free-form memory and explicit `supports` references, restricted to its current episode. Consolidator can inspect selected memories and their accessible source steps, then emits a summary with explicit parent references and citations. These are model-declared citations, not automatically proven support. A replacement may cite only tasks belonging to its declared contributing parents, which must be in the selected region.

A cited episode must still exist, its ID must match, and its step must be an integer in range; boolean/negative/out-of-range indices are invalid. All declared references must pass these checks. Summary content and applicability must be nonempty, but need not quote the source. A real citation can accompany an incorrect summary; structural admission cannot detect that.

## Source coverage and selection

`source_coverage_count(memory)` counts distinct task IDs among available, structurally valid citations. Duplicate steps or multiple memories from one task do not add coverage. It does not count retrievals, parent entries or generated QA. It is a coverage proxy, not an independent semantic support score. The regular selector and cold-output packing rank by this count, with seeded tie-breaking.

Existing storage/configuration keys `supports`, `min_support_tasks` (default 2) and `selection_policy="support"` remain for compatibility, with this structural meaning. The Python checker is now constructed as `Validator(store)`; the old injected verifier argument and `support_count` method are retired. Code callers should use `source_coverage_count`.

## Training materials and boundaries

Decision demonstrations use the original observation and action at a cited step. The decoder copies these fields and rejects supplied replacements. The source action must have been accepted or explicitly marked admissible; it need not produce immediate score gain. Material checks enforce exact source fidelity, same-step input, memory-version binding and citation scope. These checks apply to demonstrations, not to wording of the summary.

A rule QA target is copied from the summary or its declared content spans. Its source references must be a nonempty subset of the memory citations; it adds no coverage. Mixed supervision still needs exactly one rule QA and at least one valid decision example. Existing budgets, student exclusions, publication checks and retirement behavior remain in force.

There is no mandatory `observed_transition` format or native replay module. Benchmark scoring may still use its official evaluator, separately from memory admission. Log generated summaries and citations for quality audits and evaluate utility on independent tasks; admission is not evidence of truth or generalization.

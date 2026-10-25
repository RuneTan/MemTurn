# Orphan adapter archives and recovery

[English](adapter_archive_recovery.md) | [简体中文](adapter_archive_recovery.zh-CN.md)

2026-10-24: orphan candidates are copied and verified before cleanup. A complete archive and file manifest are required before any source file is removed.

## Startup recovery

A writable Runtime first recovers database job states, verifies and loads the committed model, and then examines the run's `adapters` directory. It recognizes only learner-generated version names consisting of `hf_` followed by 32 lowercase hexadecimal characters. The current model, versions referenced by active tasks, and models registered in the process are protected. Unknown directory names remain and count toward the version limit. Read-only evaluation does not perform archive recovery.

For each orphan candidate, including an incomplete save:

1. Check that the path stays inside this run's adapter directory. Reject links, Windows reparse points and directories that cannot be read completely.
2. Copy it to `research_archive/orphan_adapters/.pending_<VERSION>_<UNIQUE_ID>/payload` under the same run. Record file sizes, SHA-256 hashes, empty directories and the original path.
3. Flush files, verify the archive, rename it to its final name, and verify both the copy and the unchanged source again.
4. Remove source files only after verification succeeds. Copy, space or verification failures prevent cleanup. If cleanup stops partway through, the complete archive already exists. A later startup can archive the remaining files without overwriting the earlier complete copy.

One Runtime process must exclusively own each run directory. This is not a multiprocess write-coordination protocol, and power-loss filesystem durability has not been validated. Recovery events use the existing audit interface.

## Manual restore

From the project directory, using its Python environment:

```powershell
.\.venv-local\Scripts\python.exe -m scripts.restore_adapter_archive <ARCHIVE_DIRECTORY> <NEW_RESTORE_DIRECTORY>
```

The full file manifest and hashes are verified before and after restoration. An existing destination is never overwritten. This command restores files only: it does not change the current model pointer or promote an uncommitted candidate. Before using a historical model, check its provenance, model configuration and commit state.

## Resource and validation boundaries

Archives sit outside the active `adapters` directory. The Agent, Consolidator, retriever and trainer cannot query them, and normal execution does not automatically read them. They are offline recovery records, not additional learning memory. Archives still consume disk space, and copying needs extra space. If space is unavailable, the original remains and the existing version limit still applies. Restored files that re-enter a run must count toward state and model-version budgets. Normal reference-based model retirement is unchanged.

The inspected historical runs with state databases contained no orphan candidates. Standalone training diagnostics without databases were retained; no bulk historical cleanup was performed.

Eleven added regression checks cover capacity release and byte-for-byte restore, current/retired/registered model protection, unknown directories, copy and verification failures, source changes, unreadable directories, interrupted cleanup and retry, overwrite refusal, path escape and link rejection, and read-only evaluation. At that stage, **104 tests passed** in the project environment; the log is `tmp/archive-full-tests.log`. These validate file recovery, not learning effects.

A later addition preserves unknown versions still advertised by the remote service for deployment-owner review. Missing local registration alone never triggers remote unloading. See [cluster handoff](cluster_handoff.md) for the subsequent delivery and validation record.

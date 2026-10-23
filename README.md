# MemTurn

**Memory Internalization for Continual Agents under Resource Constraints**

MemTurn investigates how an agent can learn from its own experience over changing task streams while operating with bounded external memory, a fixed-size LoRA adapter, and explicit learning and service budgets.

**Status: research design and repository setup.** Implementation and empirical evaluation are in preparation. This repository currently contains the project overview and a literature index.

## Current design

- Acquire experience through task execution and external memory.
- Consolidate experience and selectively internalize it into a fixed-size task-model adapter.
- Manage external memory capacity while evaluating cumulative utility, learning costs, and service quality over changing task streams.

The research objective is higher cumulative utility under specified resource and service constraints. These design choices are hypotheses to evaluate, rather than established performance results.

## Repository scope

This Git repository is rooted in the `MemTurn` project folder. Only files inside that folder are selected for upload. Detailed working notes, the research memo, build handbook, downloaded papers, archives, and credentials are maintained separately in the local parent workspace.

The [literature index](docs/literature.md) links directly to original papers and is self-contained when viewed on GitHub.

## Development plan

1. **P0–P1:** select the model and hardware, then build a small task environment with checkable feedback.
2. **P2–P3:** implement retrieval, writing, consolidation, provenance checks, and capacity maintenance.
3. **P4–P5:** validate one-memory SFT and continued updates to the same fixed-size LoRA.
4. **P6:** implement background jobs, consistent version publication, and recovery while serving tasks.
5. **P7:** evaluate long task streams against matched-budget baselines.

Dependencies and executable commands will be documented as the implementation becomes available.

## Working with the repository

GitHub hosts the source and its history. Model inference, SFT, and experiments run on a local machine or GPU server.

In the existing local workspace, run Git commands from the `MemTurn` subfolder. Files outside this checkout do not enter its commits.

To obtain a working copy on another machine:

```powershell
git clone https://github.com/RuneTan/MemTurn-Memory-Internalization-for-Continual-Agents-under-Resource-Constraints.git MemTurn
cd MemTurn
```

Authenticate to GitHub first if repository access requires it. Repository visibility and permissions are managed on GitHub.

After making changes, review them and commit only the intended files:

```powershell
git status
git diff
git add path/to/changed-file
git commit -m "Describe the change"
git push
```

Use separate feature branches once implementation begins. Before starting work on another machine, update that machine's working copy when its local changes have been committed. Store credentials in a local `.env` file or an appropriate secret store; runtime data, downloaded weights, and experiment outputs stay outside Git history.

## Literature and attribution

The proposed system builds on existing research; see the [literature index](docs/literature.md). The current documents do not establish novelty, successful reproduction, or experimental advantages. A software license has not yet been selected.

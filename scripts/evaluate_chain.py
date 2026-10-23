"""Local-only paired end-state task probes; no Writer calls or training."""
import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def evaluate(run, show_tool_state=False, seed_base=1000000):
    from memturn.config import Config
    from memturn.contracts import Artifact, Task, dumps
    from memturn.memory.store import Store
    from memturn.models.http import ChatPolicy
    from memturn.models.local import LocalHFChat, LocalModelConfig
    from memturn.runtime.service import Runtime
    from memturn.tasks.toy import PREPARE, ToyEnvironment
    from memturn.tokens import HuggingFaceCounter
    spec = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    if report["reason"] == "running":
        raise ValueError("wait for the chain to pause before final evaluation")
    settings = spec["profile"]["model"]
    backend = LocalHFChat(LocalModelConfig(str(ROOT / settings["path"]), settings["device"],
                                          settings["dtype"], settings["max_input_tokens"]))
    counter = HuggingFaceCounter(settings["id"], settings["revision"], backend.tokenizer)
    store = Store(run / "state.sqlite", Config(**spec["profile"]["runtime"]), counter)
    before = store.stats()
    if store.artifact:
        artifact = Artifact(**store.artifact)
        for name, checksum in artifact.metrics["checksums"].items():
            if hashlib.sha256((Path(artifact.path) / name).read_bytes()).hexdigest() != checksum:
                raise ValueError("adapter checksum mismatch")
        backend.load(artifact)
    policy = ChatPolicy(backend.role("final_evaluation", settings["task_max_new_tokens"]))
    class ForcedPolicy:
        def __init__(self, version, use_memory):
            self.version, self.use_memory = version, use_memory
            self.search_requests, self.last_query = 0, None
            self.first_decision = None
        def act(self, observation, memories, version="base", history=()):
            selected_memories = memories if self.use_memory else ()
            if show_tool_state:
                state = json.loads(observation)
                state["controller_tool_status"] = {
                    "memory_search_remaining": max(0, store.config.max_extra_searches - self.search_requests),
                    "previous_requested_query": self.last_query,
                    "current_result_refs": [m.ref for m in selected_memories],
                    "instruction": "A previous search has completed. Do not repeat it without new information. "
                                   "When remaining is zero, choose an environment command instead of memory_search."}
                observation = dumps(state)
            action = policy.act(observation, selected_memories, self.version, history)
            if self.first_decision is None:
                self.first_decision = action
            if isinstance(action, dict) and action.get("kind") == "memory_search":
                self.search_requests += 1
                self.last_query = action.get("query")
            return action
    class DisabledWriter:
        def write(self, *args):
            raise AssertionError("evaluation must not write experience")
    runtime = Runtime(store, DisabledWriter(), None, None)
    rows = []
    try:
        for label, version, use_memory in (("base_no_memory", "base", False),
                                          ("adapter_no_memory", store.model_version, False),
                                          ("base_current_memory", "base", True),
                                          ("adapter_current_memory", store.model_version, True)):
            for condition_index, condition in enumerate(PREPARE):
                family, version_label = condition.split(":")
                for sample in range(2):
                    task = Task(f"final_{label}_{condition_index}_{sample}", family, version_label,
                                seed_base + condition_index * 10 + sample,
                                metadata={"evaluation_only": True, "memory_enabled": use_memory})
                    forced = ForcedPolicy(version, use_memory)
                    episode = runtime.run_task(task, ToyEnvironment(), forced)
                    episode = replace(episode, model_version=version)
                    row = {"regime": label, "condition": condition, "sample": sample,
                           "tool_state_visible": show_tool_state,
                           "seed_base": seed_base, "first_decision": forced.first_decision,
                           "first_decision_correct": forced.first_decision == PREPARE[condition],
                           "success": episode.success, "reward": episode.reward,
                           "steps": len(episode.steps), "status": episode.status,
                           "actions": [s.action for s in episode.steps],
                           "accepted": [s.accepted for s in episode.steps], "trace": episode.to_dict()}
                    rows.append(row)
                    print(dumps({k: row[k] for k in ("regime", "condition", "sample", "success", "reward")}), flush=True)
        runtime.drain()
        if store.stats() != before:
            raise AssertionError("final evaluation changed the recorded learning state")
    finally:
        runtime.close()
        store.close()
    output = run / "analysis"
    output.mkdir(exist_ok=True)
    stem = "final_evaluation_tools_visible" if show_tool_state else "final_evaluation"
    if seed_base != 1000000:
        stem += f"_seed{seed_base}"
    (output / (stem + ".jsonl")).write_text("".join(dumps(r) + "\n" for r in rows), encoding="utf-8")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["regime"]].append(row)
    summary = {name: {"tasks": len(values), "success_rate": sum(r["success"] for r in values) / len(values),
                      "first_decision_accuracy": sum(r["first_decision_correct"] for r in values) / len(values),
                      "mean_reward": sum(r["reward"] for r in values) / len(values)}
               for name, values in grouped.items()}
    (output / (stem + "_summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--show-tool-state", action="store_true")
    parser.add_argument("--seed-base", type=int, default=1000000)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run.resolve(), args.show_tool_state, args.seed_base), indent=2))

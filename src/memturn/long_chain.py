"""Exploratory grouped stream with clean-boundary resume and held-out diagnostics."""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from .chain_support import BudgetedChat, ResumableWriter, grouped_tasks, probe_actions
from .config import Config
from .contracts import Episode, dumps
from .learning.huggingface import HuggingFaceLearner, SFTConfig
from .memory.consolidation import ToyConsolidator
from .memory.store import Store
from .memory.validation import Validator
from .models.agents import ChatWriter
from .models.http import ChatConfig, ChatPolicy, HttpChat
from .models.local import LocalHFChat, LocalModelConfig
from .runtime.service import Runtime
from .tasks.toy import PREPARE, ToyEnvironment, ToyVerifier
from .tokens import HuggingFaceCounter


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class ObservedPolicy:
    def __init__(self, policy):
        self.policy, self.refs, self.conditions = policy, set(), set()
        self.first_action = None

    def act(self, observation, memories, model_version="base", history=()):
        self.refs.update(m.ref for m in memories)
        self.conditions.update(m.applicability for m in memories)
        action = self.policy.act(observation, memories, model_version, history)
        if self.first_action is None and isinstance(action, str):
            self.first_action = action
        return action


def run_chain(profile_path, writer_profile, output, tasks=500, group_size=20,
              writer_token_budget=180000, resume=False):
    import torch
    import psutil
    project = Path(__file__).resolve().parents[2]
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    writer_config = json.loads(Path(writer_profile).read_text(encoding="utf-8"))
    if not os.environ.get(writer_config["api_key_env"]):
        raise ValueError("set the named Writer API key environment variable first")
    settings = profile["model"]
    model_path = (project / settings["path"]).resolve()
    source = json.loads((model_path / "memturn_source.json").read_text(encoding="utf-8"))
    if source != {"model_id": settings["id"], "revision": settings["revision"]}:
        raise ValueError("local model source mismatch")
    os.environ.setdefault("HF_HOME", str(project / ".cache" / "huggingface"))
    spec = {"profile": profile, "writer": writer_config, "tasks": tasks,
            "group_size": group_size, "writer_token_budget": writer_token_budget,
            "protocol_version": "retrieval-transcript-v3-sft-coverage-v2",
            "scope": "exploratory_grouped_toy_stream_not_formal_benchmark"}
    fingerprint = hashlib.sha256(dumps(spec).encode()).hexdigest()
    root = Path(output).resolve()
    checkpoint = {}
    if resume:
        checkpoint = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
        if checkpoint["fingerprint"] != fingerprint:
            raise ValueError("resume settings or protocol differ from the frozen experiment; keep the old run and start a new one")
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "experiment.json", spec)
        with (root / "task_manifest.jsonl").open("w", encoding="utf-8") as stream:
            for task in grouped_tasks(tasks, group_size, profile["runtime"]["seed"]):
                stream.write(dumps(asdict(task)) + "\n")
    counts = Counter(checkpoint.get("events", {}))
    audit_lock = threading.Lock()
    started = time.monotonic()
    elapsed_before = checkpoint.get("elapsed_seconds", 0)
    with (root / "audit.jsonl").open("a", encoding="utf-8") as log:
        def audit(event):
            with audit_lock:
                counts[event["event"]] += 1
                log.write(dumps(dict(event, time=time.time())) + "\n")
                log.flush()
        torch.manual_seed(profile["runtime"]["seed"])
        backend = LocalHFChat(LocalModelConfig(str(model_path), settings["device"], settings["dtype"],
                                              settings["max_input_tokens"]), audit)
        counter = HuggingFaceCounter(settings["id"], settings["revision"], backend.tokenizer)
        config = Config(**profile["runtime"])
        store = Store(root / "state.sqlite", config, counter)
        if store.stats()["completed"] != checkpoint.get("next_index", 0):
            store.close()
            raise ValueError("unclean stop outside a saved boundary; preserve files for explicit recovery")
        values = dict(profile["training"], model_id=str(model_path), model_revision=settings["revision"],
                      device=settings["device"], dtype=settings["dtype"])
        values["target_modules"] = tuple(values["target_modules"])
        learner = HuggingFaceLearner(root / "adapters", SFTConfig(**values), backend)
        client = BudgetedChat(HttpChat(ChatConfig(**writer_config), audit), audit,
                              writer_token_budget, checkpoint.get("writer_usage"))
        guidance = ("For this development environment, extract the successful PREPARATION action, "
                    "not the terminal open/retry action. Keep its exact command name in content. "
                    "applicability must exactly equal the source condition label such as vault:v1. "
                    "Cite only zero-based accepted preparation steps; rejected actions are not support. "
                    "Consolidate different tasks only when their condition and preparation rule match.")
        writer = ResumableWriter(ChatWriter(client, guidance))
        runtime = Runtime(store, writer, ToyConsolidator(store), Validator(store, ToyVerifier()), learner, audit)
        policy = ChatPolicy(backend.role("task", settings["task_max_new_tokens"]))
        probe_policy = ChatPolicy(backend.role("probe", settings["task_max_new_tokens"]))
        process = psutil.Process()

        def save(reason):
            completed = store.stats()["completed"]
            data = {"fingerprint": fingerprint, "target_tasks": tasks, "next_index": completed,
                    "reason": reason, "events": dict(counts), "writer_usage": dict(client.usage),
                    "elapsed_seconds": elapsed_before + time.monotonic() - started,
                    "deferred_writer": writer.deferred.to_dict() if writer.deferred else None}
            write_json(root / "checkpoint.json", data)
            report = dict(data, scope=spec["scope"], stats=store.stats(), errors=runtime.errors,
                          writer_charged_tokens=client.charged_tokens,
                          session_writer_tokens=client.budget_used,
                          gpu=torch.cuda.get_device_name(), model_id=settings["id"],
                          group_size=group_size, arrival_policy="wait_for_maintenance",
                          operator_kind="Groq_Writer_fixture_Consolidator_real_local_LoRA",
                          output=str(root))
            write_json(root / "report.json", report)

        def probe(completed):
            rows = probe_actions(backend, store, probe_policy, counter, config, completed)
            with (root / "probes.jsonl").open("a", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(dumps(row) + "\n")
            write_json(root / f"bank_{completed:04}.json", [m.to_dict() for m in store.memories()])
            print(dumps({"event": "probe", "completed": completed,
                         "accuracy": {name: sum(r["correct"] for r in rows if r["regime"] == name) / 8
                                      for name in sorted({r["regime"] for r in rows})}}), flush=True)

        try:
            if checkpoint.get("deferred_writer"):
                episode = Episode.from_dict(checkpoint["deferred_writer"])
                store.requeue_source(episode)
                runtime.queue.put(episode)
                runtime.drain()
                if writer.deferred:
                    save(client.paused)
                    return json.loads((root / "report.json").read_text(encoding="utf-8"))
            if not resume:
                probe(0)
                save("running")
            begin = store.stats()["completed"]
            reason = "target_completed"
            with (root / "metrics.jsonl").open("a", encoding="utf-8") as metrics:
                for index, task in enumerate(grouped_tasks(tasks, group_size, config.seed)):
                    if index < begin:
                        continue
                    if not client.can_start():
                        reason = "configured_writer_token_budget"
                        break
                    torch.cuda.reset_peak_memory_stats()
                    observed = ObservedPolicy(policy)
                    before = store.model_version
                    episode = runtime.run_task(task, ToyEnvironment(), observed)
                    runtime.drain()
                    stats = store.stats()
                    row = {"index": index, "completed": index + 1, "group": index // group_size,
                           "condition": episode.condition, "success": episode.success,
                           "reward": episode.reward, "steps": len(episode.steps), "status": episode.status,
                           "first_action": observed.first_action,
                           "first_action_correct": observed.first_action == PREPARE[episode.condition],
                           "retrieved_refs": sorted(observed.refs), "retrieved_conditions": sorted(observed.conditions),
                           "model_before": before, "model_after": store.model_version,
                           "stats": stats, "events": dict(counts), "writer_usage": dict(client.usage),
                           "elapsed_seconds": elapsed_before + time.monotonic() - started,
                           "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                           "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                           "rss_after_maintenance_bytes": process.memory_info().rss,
                           "adapter_files": len(list((root / "adapters").rglob("*.safetensors")))}
                    metrics.write(dumps(row) + "\n")
                    metrics.flush()
                    if writer.deferred:
                        reason = client.paused
                        save(reason)
                        break
                    if (index + 1) % group_size == 0:
                        probe(index + 1)
                    save("running")
                    if (index + 1) % 5 == 0:
                        print(dumps({"event": "progress", "completed": index + 1,
                                     "condition": episode.condition, "utility": stats["utility"],
                                     "memories": stats["memories"], "active_tokens": stats["active_tokens"],
                                     "publications": counts["adapter_published"],
                                     "writer_tokens": client.charged_tokens}), flush=True)
            completed = store.stats()["completed"]
            if completed % group_size and not writer.deferred:
                probe(completed)
            save(reason)
            return json.loads((root / "report.json").read_text(encoding="utf-8"))
        finally:
            runtime.close()
            store.close()

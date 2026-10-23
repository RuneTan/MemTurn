"""Run the ordinary runtime with local LM calls and optional genuine LM LoRA SFT."""
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import threading
import time

from .config import Config
from .contracts import dumps
from .learning.huggingface import HuggingFaceLearner, SFTConfig
from .memory.consolidation import ToyConsolidator, ToyWriter
from .memory.store import Store
from .memory.validation import Validator
from .models.agents import ChatConsolidator, ChatWriter
from .models.http import ChatConfig, ChatPolicy, HttpChat
from .models.local import LocalHFChat, LocalModelConfig
from .runtime.service import Runtime
from .tasks.toy import ToyEnvironment, ToyVerifier, task_stream


def run_local(profile_path, tasks=8, output=None, train=False, operators="model", writer_profile=None):
    import torch
    from .tokens import HuggingFaceCounter
    profile_path = Path(profile_path).resolve()
    project = profile_path.parent.parent
    os.environ.setdefault("HF_HOME", str(project / ".cache" / "huggingface"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    settings = profile["model"]
    model_path = (project / settings["path"]).resolve()
    source = json.loads((model_path / "memturn_source.json").read_text(encoding="utf-8"))
    if source != {"model_id": settings["id"], "revision": settings["revision"]}:
        raise ValueError("downloaded model source does not match the pinned local profile")
    writer_client = None
    if writer_profile:
        writer_settings = json.loads(Path(writer_profile).read_text(encoding="utf-8"))
        chat_config = ChatConfig(**writer_settings)
        if not os.environ.get(chat_config.api_key_env):
            raise ValueError(f"Set {chat_config.api_key_env} before running the remote Writer")
        writer_client = HttpChat(chat_config)
    root = Path(output).resolve() if output else project / "runs" / ("local_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    root.mkdir(parents=True, exist_ok=False)
    (root / "profile.json").write_text(dumps(profile), encoding="utf-8")
    counts, audit_lock = Counter(), threading.Lock()
    started = time.monotonic()
    with (root / "audit.jsonl").open("w", encoding="utf-8") as stream:
        def audit(event):
            with audit_lock:
                counts[event["event"]] += 1
                stream.write(dumps(event) + "\n")
                stream.flush()
        backend = LocalHFChat(LocalModelConfig(str(model_path), settings["device"], settings["dtype"],
                                              settings["max_input_tokens"]), audit)
        # Reuse the already loaded tokenizer; no extra network or cache lookup.
        counter = HuggingFaceCounter(settings["id"], settings["revision"], tokenizer=backend.tokenizer)
        store = Store(root / "state.sqlite", Config(**profile["runtime"]), counter)
        learner = None
        if train:
            cfg = dict(profile["training"], model_id=str(model_path), model_revision=settings["revision"],
                       device=settings["device"], dtype=settings["dtype"])
            cfg["target_modules"] = tuple(cfg["target_modules"])
            learner = HuggingFaceLearner(root / "adapters", SFTConfig(**cfg), backend)
        guidance = ("For this development environment, extract the successful PREPARATION action, "
                    "not the terminal open/retry action. Keep its exact command name in content. "
                    "applicability must exactly equal the source condition label such as vault:v1. "
                    "Cite only zero-based accepted preparation steps; rejected actions are not support. "
                    "Consolidate different tasks only when their condition and preparation rule match.")
        writer = ChatWriter(backend.role("writer", settings["operator_max_new_tokens"]), guidance) if operators == "model" else ToyWriter()
        if writer_client is not None:
            writer = ChatWriter(writer_client, guidance)
        consolidator = ChatConsolidator(backend.role("consolidator", settings["operator_max_new_tokens"]), store,
                                       counter, max_turns=8, task_guidance=guidance) if operators == "model" else ToyConsolidator(store)
        runtime = Runtime(store, writer, consolidator, Validator(store, ToyVerifier()), learner, audit)
        outcomes = []
        try:
            for task in task_stream(tasks, store.config.seed):
                result = runtime.run_task(task, ToyEnvironment(), ChatPolicy(backend.role("task", settings["task_max_new_tokens"])))
                outcomes.append({"task_id": result.task_id, "success": result.success,
                                 "reward": result.reward, "status": result.status})
                runtime.drain()
            report = {"scope": "local_lm_engineering_not_benchmark", "model_id": settings["id"],
                      "model_revision": settings["revision"], "operators": operators, "training": train,
                      "arrival_policy": "wait_for_background_between_tasks", "stats": store.stats(),
                      "events": dict(counts), "errors": runtime.errors, "outcomes": outcomes,
                      "model_usage": backend.usage, "elapsed_seconds": time.monotonic() - started,
                      "writer_api": {"model": writer_client.config.model, "usage": writer_client.usage} if writer_client else None,
                      "gpu": torch.cuda.get_device_name(),
                      "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                      "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "accelerate")},
                      "output": str(root)}
            (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return report
        finally:
            runtime.close()
            store.close()

"""Remote assembly: model, task, verifier and trainer are independently replaceable."""
from collections import Counter
import importlib
import json
import os
from pathlib import Path

from .config import Config
from .contracts import Task, dumps
from .learning.huggingface import HuggingFaceLearner, SFTConfig
from .memory.store import Store
from .memory.validation import Validator
from .models.agents import ChatConsolidator, ChatWriter
from .models.http import ChatConfig, ChatPolicy, HttpChat
from .models.publication import VLLMAdapters
from .runtime.service import Runtime
from .tokens import HuggingFaceCounter


def inspect_profile(profile):
    missing = []
    def visit(value, path=""):
        if isinstance(value, dict):
            for key, child in value.items(): visit(child, f"{path}.{key}".lstrip("."))
        elif isinstance(value, list):
            for index, child in enumerate(value): visit(child, f"{path}[{index}]")
        elif isinstance(value, str) and value.startswith("TO_"):
            missing.append(path)
    visit(profile)
    Config(**profile["runtime"])
    if profile.get("training", {}).get("enabled") and not profile["training"].get("target_modules"):
        missing.append("training.target_modules")
    for role, settings in profile["models"].items():
        if not os.environ.get(settings["base_url_env"]):
            missing.append(f"models.{role}.base_url_env:{settings['base_url_env']}")
    return sorted(set(missing))


def environment_for(settings):
    if settings.get('worker_command'):
        from .tasks.process import ProcessEnvironment
        return ProcessEnvironment(settings)
    if settings["kind"] == "scienceworld":
        from .tasks.scienceworld import ScienceWorldEnvironment
        return ScienceWorldEnvironment(jar_path=settings.get("jar_path"),
                                       simplifications=settings.get("simplifications", ""))
    if settings["kind"] == "alfworld":
        from .tasks.alfworld import ALFWorldEnvironment
        return ALFWorldEnvironment(settings["config_path"], settings["split"])
    raise ValueError("Only ScienceWorld and ALFWorld are in the first benchmark phase")


def run_profile(profile_path, output):
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    missing = inspect_profile(profile)
    if missing:
        raise ValueError("Complete remote setup first: " + ", ".join(missing))
    module, name = profile["verification_factory"].split(":", 1)
    verifier = getattr(importlib.import_module(module), name)()
    if verifier.__class__.__name__ == "BenchmarkVerifierRequired":
        raise ValueError("A benchmark-specific evidence verifier must be implemented and validated")
    clients = {role: HttpChat(ChatConfig(os.environ[cfg["base_url_env"]], cfg["model"],
                cfg["api_key_env"], cfg.get("timeout_seconds", 60), cfg.get("max_tokens", 2048),
                chat_template_kwargs=cfg.get("chat_template_kwargs"), response_format=cfg.get("response_format")))
               for role, cfg in profile["models"].items()}
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "state.sqlite").exists():
        raise ValueError("Use a new run directory; explicit remote resume is not enabled yet")
    (root / "profile.json").write_text(dumps(profile), encoding="utf-8")
    counter = HuggingFaceCounter(profile["tokenizer_model"], profile["tokenizer_revision"])
    config = Config(**profile["runtime"])
    store = Store(root / "state.sqlite", config, counter)
    learner = None
    if profile["training"]["enabled"]:
        settings = profile["training"]
        adapter_root = root / "adapters"
        serving = VLLMAdapters(os.environ[profile["models"]["task"]["base_url_env"]], adapter_root,
                               settings["server_adapter_root"], profile["models"]["task"]["api_key_env"])
        values = {key: value for key, value in settings.items() if key not in {"enabled", "server_adapter_root"}}
        values["target_modules"] = tuple(values["target_modules"])
        learner = HuggingFaceLearner(adapter_root, SFTConfig(**values), serving)
    counts = Counter()
    with (root / "audit.jsonl").open("w", encoding="utf-8") as stream:
        import threading
        audit_lock = threading.Lock()
        def audit(event):
            with audit_lock:
                counts[event["event"]] += 1
                stream.write(dumps(event) + "\n")
                stream.flush()
        runtime = Runtime(store, ChatWriter(clients["writer"]),
            ChatConsolidator(clients["consolidator"], store, counter), Validator(store, verifier), learner, audit)
        try:
            with Path(profile["environment"]["task_manifest"]).open(encoding="utf-8") as manifest:
                for line in manifest:
                    if line.strip():
                        task = Task(**json.loads(line))
                        runtime.run_task(task, environment_for(profile["environment"]), ChatPolicy(clients["task"]))
                        if profile.get("arrival_policy") == "wait_for_background":
                            runtime.drain()
            runtime.drain()
            report = {"stats": store.stats(), "events": dict(counts), "errors": runtime.errors,
                      "model_usage": {role: client.usage for role, client in clients.items()},
                      "operator_kind": profile["operator_kind"], "arrival_policy": profile["arrival_policy"]}
            (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            return report
        finally:
            runtime.close()
            store.close()

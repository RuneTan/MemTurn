"""Offline commands have no network calls and require no model downloads."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import threading

from .config import Config
from .contracts import dumps
from .memory.consolidation import ToyConsolidator, ToyWriter
from .memory.store import Store
from .memory.validation import Validator
from .runtime.service import Runtime
from .tasks.toy import ToyEnvironment, ToyPolicy, ToyVerifier, task_stream
from .tokens import OfflineCounter


def demo(args):
    config = Config.load(args.config) if args.config else Config()
    output = Path(args.output or ("runs/demo_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "state.sqlite").exists():
        raise SystemExit("Use a new output directory; existing runs are preserved.")
    (output / "config.json").write_text(dumps(config.to_dict()), encoding="utf-8")
    counter = OfflineCounter()
    store = Store(output / "state.sqlite", config, counter)
    learner = None
    if args.train_tiny:
        from .learning.tiny_lora import TinyLora
        learner = TinyLora(output / "adapters")
    audit_lock, counts = threading.Lock(), Counter()
    with (output / "audit.jsonl").open("w", encoding="utf-8") as log:
        def audit(event):
            with audit_lock:
                counts[event["event"]] += 1
                log.write(dumps(event) + "\n")
                log.flush()
        runtime = Runtime(store, ToyWriter(), ToyConsolidator(store), Validator(store, ToyVerifier()), learner, audit)
        try:
            for task in task_stream(args.tasks, config.seed):
                runtime.run_task(task, ToyEnvironment(), ToyPolicy(learner))
                runtime.drain()
            report = {"scope": "offline_engineering_smoke_not_benchmark_results",
                      "mode": "tiny_action_lora" if learner else "memory_only",
                      "arrival_policy": "wait_for_background_between_tasks",
                      "stats": store.stats(), "events": dict(counts), "errors": runtime.errors,
                      "output": str(output)}
            (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=2))
        finally:
            runtime.close()
            store.close()


def doctor(args):
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "alfworld", "scienceworld", "gymnasium"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    print(json.dumps({"python": platform.python_version(), "platform": platform.platform(),
                      "packages": packages, "network_checks": "not_run"}, indent=2))


def check_profile(args):
    from .experiments import inspect_profile
    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    missing = inspect_profile(profile)
    print(json.dumps({"ready": not missing, "unresolved": missing,
                      "network_checks": "not_run", "operator_kind": profile["operator_kind"]}, indent=2))


def run_remote(args):
    from .experiments import run_profile
    print(json.dumps(run_profile(args.profile, args.output), ensure_ascii=False, indent=2))


def local_smoke(args):
    from .local_experiment import run_local
    print(json.dumps(run_local(args.profile, args.tasks, args.output, args.train, args.operators, args.writer_profile),
                     ensure_ascii=False, indent=2))


def long_chain(args):
    from .long_chain import run_chain
    print(json.dumps(run_chain(args.profile, args.writer_profile, args.output, args.tasks,
                               args.group_size, args.writer_token_budget, args.resume),
                     ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(prog="memturn")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("demo", help="run the offline two-family engineering fixture")
    smoke.add_argument("--config")
    smoke.add_argument("--output")
    smoke.add_argument("--tasks", type=int, default=24)
    smoke.add_argument("--train-tiny", action="store_true", help="real CPU LoRA gradients on a tiny action classifier")
    smoke.set_defaults(handler=demo)
    check = sub.add_parser("doctor", help="report dependencies without exposing credentials")
    check.set_defaults(handler=doctor)
    profile = sub.add_parser("check-profile", help="list unresolved remote settings without connecting")
    profile.add_argument("--profile", required=True)
    profile.set_defaults(handler=check_profile)
    remote = sub.add_parser("run", help="execute a completed remote benchmark profile")
    remote.add_argument("--profile", required=True)
    remote.add_argument("--output", required=True)
    remote.set_defaults(handler=run_remote)
    local = sub.add_parser("local-smoke", help="run a downloaded local LM in the engineering environment")
    local.add_argument("--profile", default="configs/local_qwen3_1_7b.json")
    local.add_argument("--tasks", type=int, default=8)
    local.add_argument("--output")
    local.add_argument("--train", action="store_true", help="genuine language-model LoRA SFT")
    local.add_argument("--operators", choices=("model", "fixture"), default="model")
    local.add_argument("--writer-profile", help="optional separate HTTP Writer config; reads only its named key env var")
    local.set_defaults(handler=local_smoke)
    chain = sub.add_parser("long-chain", help="grouped exploratory local stream with Writer budget and probes")
    chain.add_argument("--profile", default="configs/local_qwen3_1_7b.json")
    chain.add_argument("--writer-profile", default="configs/writer.groq.example.json")
    chain.add_argument("--output", required=True)
    chain.add_argument("--tasks", type=int, default=500)
    chain.add_argument("--group-size", type=int, default=20)
    chain.add_argument("--writer-token-budget", type=int, default=180000)
    chain.add_argument("--resume", action="store_true")
    chain.set_defaults(handler=long_chain)
    args = parser.parse_args()
    if getattr(args, "tasks", 1) <= 0:
        parser.error("tasks must be positive")
    try:
        args.handler(args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"memturn: {error}\n")

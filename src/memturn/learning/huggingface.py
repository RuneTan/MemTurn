"""Remote-worker SFT implementation; optional dependencies are imported lazily."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
import uuid

from ..contracts import Artifact, dumps
from ..models.http import environment_action, task_messages


def supervised_tokens(tokenizer, prompt, target, max_length, memories=(), history=()):
    """Reject unstable template prefixes and overlong samples; never truncate targets."""
    messages = task_messages(prompt, memories, history)
    kwargs = {"tokenize": False, "enable_thinking": False}
    prefix = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": environment_action(target)}],
                                         add_generation_prompt=False, **kwargs)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full_ids = tokenizer.encode(full, add_special_tokens=False)
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("chat template prefix differs between prompt and target; configure template explicitly")
    if len(full_ids) > max_length or len(prefix_ids) >= len(full_ids):
        raise ValueError("sample is overlong or has no assistant target tokens")
    return full_ids, [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]


def training_schedule(size, steps, accumulation, seed):
    """Cover every sample before repeats, within the requested optimizer steps."""
    if min(size, steps, accumulation) <= 0:
        raise ValueError("training schedule requires positive sizes")
    effective = max(accumulation, math.ceil(size / steps))
    order, rng = [], random.Random(seed)
    while len(order) < steps * effective:
        epoch = list(range(size))
        rng.shuffle(epoch)
        order.extend(epoch)
    return tuple(tuple(order[i:i + effective]) for i in range(0, steps * effective, effective))


@dataclass(frozen=True)
class SFTConfig:
    model_id: str
    model_revision: str
    target_modules: tuple[str, ...]
    loader_class: str = "AutoModelForMultimodalLM"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    rank: int = 8
    alpha: int = 16
    learning_rate: float = 5e-5
    steps: int = 8
    gradient_accumulation: int = 4
    max_length: int = 2048
    max_training_tokens: int = 16384
    max_seconds: float = 1200.0
    seed: int = 42

    def __post_init__(self):
        for name in ("rank", "alpha", "learning_rate", "steps", "gradient_accumulation",
                     "max_length", "max_training_tokens", "max_seconds"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


class HuggingFaceLearner:
    accepts_memory_context = True

    def __init__(self, directory, config, serving):
        if not config.model_revision or config.model_revision.startswith("TO_") or not config.target_modules:
            raise ValueError("freeze the actual model revision and verified LoRA target modules first")
        self.root, self.config, self.serving = Path(directory).resolve(), config, serving
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts = {}

    def train(self, examples, parent_version, memory_context=(), histories=None):
        import torch
        import transformers
        from peft import LoraConfig, PeftModel, get_peft_model
        cfg, started = self.config, time.monotonic()
        tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.model_revision)
        batches, tokens = [], 0
        examples = tuple(examples)
        for example in examples:
            history = (histories or {}).get((example.task_id, example.step), ())
            ids, labels = supervised_tokens(tokenizer, example.prompt, example.target, cfg.max_length,
                                           memory_context, history)
            if tokens + len(ids) > cfg.max_training_tokens:
                raise ValueError("training token budget cannot cover all selected examples; memory must be retained")
            batches.append((ids, labels))
            tokens += len(ids)
        if not batches:
            raise ValueError("no valid SFT data fits the token budget")
        allowed = {"AutoModelForCausalLM", "AutoModelForMultimodalLM", "AutoModelForImageTextToText"}
        if cfg.loader_class not in allowed:
            raise ValueError("unsupported model loader")
        loader = getattr(transformers, cfg.loader_class)
        model = loader.from_pretrained(cfg.model_id, revision=cfg.model_revision,
                                      torch_dtype=getattr(torch, cfg.dtype)).to(cfg.device)
        if parent_version == "base":
            model = get_peft_model(model, LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha,
                target_modules=list(cfg.target_modules), bias="none", task_type="CAUSAL_LM"))
        else:
            parent = self.artifacts.get(parent_version)
            if not parent:
                raise ValueError("current adapter must be registered before continuation")
            if parent.metrics["rank"] != cfg.rank or parent.metrics["model_revision"] != cfg.model_revision:
                raise ValueError("adapter rank and base-model revision must remain fixed")
            model = PeftModel.from_pretrained(model, parent.path, is_trainable=True)
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not trainable or any("lora_" not in name for name, _ in trainable):
            raise ValueError("only LoRA parameters may be trainable")
        before = [p.detach().cpu().clone() for _, p in trainable]
        optimizer = torch.optim.AdamW([p for _, p in trainable], lr=cfg.learning_rate)
        losses, processed_tokens, consumed = [], 0, []
        schedule = training_schedule(len(batches), cfg.steps, cfg.gradient_accumulation, cfg.seed)
        model.train()
        for step, indices in enumerate(schedule):
            if time.monotonic() - started >= cfg.max_seconds:
                break
            microbatches = []
            for index in indices:
                ids, labels = batches[index]
                if processed_tokens + len(ids) > cfg.max_training_tokens:
                    break
                microbatches.append((index, ids, labels))
                processed_tokens += len(ids)
            if not microbatches:
                break
            optimizer.zero_grad()
            step_loss = 0.0
            for index, ids, labels in microbatches:
                if time.monotonic() - started >= cfg.max_seconds:
                    raise RuntimeError("training deadline reached before the planned optimizer update")
                inputs = torch.tensor([ids], device=cfg.device)
                output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                               labels=torch.tensor([labels], device=cfg.device))
                if not torch.isfinite(output.loss):
                    raise RuntimeError("non-finite language-model SFT loss")
                (output.loss / len(microbatches)).backward()
                step_loss += float(output.loss.detach()) / len(microbatches)
                example = examples[index]
                consumed.append({"index": index, "task_id": example.task_id, "source_step": example.step,
                                 "input_step": example.step if example.input_step is None else example.input_step,
                                 "optimizer_step": step, "tokens": len(ids),
                                 "input_sha256": hashlib.sha256(example.prompt.encode()).hexdigest()})
            if any(p.grad is not None for _, p in model.named_parameters() if not p.requires_grad):
                raise RuntimeError("frozen base received gradients")
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(step_loss)
        if {row["index"] for row in consumed} != set(range(len(batches))):
            raise RuntimeError("incomplete training sample coverage; memory must be retained")
        if not losses or not any(not torch.equal(old, p.detach().cpu()) for old, (_, p) in zip(before, trainable)):
            raise RuntimeError("no optimizer update reached the adapter")
        version = "hf_" + uuid.uuid4().hex
        path = self.root / version
        model.save_pretrained(path, safe_serialization=True)
        checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()}
        metrics = {"steps": len(losses), "losses": losses, "unique_training_tokens": tokens,
                   "processed_training_tokens": processed_tokens,
                   "validated_examples": len(examples), "unique_consumed_examples": len({r['index'] for r in consumed}),
                   "sample_coverage": 1.0, "consumed_examples": consumed,
                   "configured_gradient_accumulation": cfg.gradient_accumulation,
                   "effective_gradient_accumulation": len(schedule[0]), "sampling_seed": cfg.seed,
                   "trainable_parameters": sum(p.numel() for _, p in trainable), "base_frozen": True,
                   "model_id": cfg.model_id, "model_revision": cfg.model_revision,
                   "rank": cfg.rank, "checksums": checksums, "seconds": time.monotonic()-started}
        artifact = Artifact(version, str(path), "huggingface_lora", parent_version, metrics)
        (path / "manifest.json").write_text(dumps(artifact.__dict__), encoding="utf-8")
        self.artifacts[version] = artifact
        del optimizer, model
        if cfg.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return artifact

    def prepare(self, artifact):
        path = Path(artifact.path).resolve()
        if path.parent != self.root:
            raise ValueError("adapter path is outside its configured root")
        for name, checksum in artifact.metrics["checksums"].items():
            if Path(name).name != name or hashlib.sha256((path/name).read_bytes()).hexdigest() != checksum:
                raise ValueError("adapter checksum mismatch")
        self.serving.load(artifact)
        self.artifacts[artifact.version] = artifact

    def retain(self, versions):
        for version, artifact in list(self.artifacts.items()):
            if version not in versions:
                self.serving.unload(version)
                path = Path(artifact.path).resolve()
                if path.parent != self.root:
                    raise ValueError("refusing cleanup outside artifact root")
                for file in path.iterdir():
                    if not file.is_file():
                        raise ValueError("unexpected directory in adapter artifact")
                    file.unlink()
                path.rmdir()
                self.artifacts.pop(version)

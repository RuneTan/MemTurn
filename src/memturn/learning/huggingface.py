"""Remote-worker SFT implementation; optional dependencies are imported lazily."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time
import uuid

from ..contracts import Artifact, dumps
from ..models.http import environment_action
from .materials import training_messages


def supervised_tokens(tokenizer, prompt, target, max_length, memories=(), history=(), output_kind='environment_action'):
    """Reject unstable template prefixes and overlong samples; never truncate targets."""
    messages = training_messages(prompt, memories, history, output_kind)
    target_text = environment_action(target) if output_kind == 'environment_action' else target
    kwargs = {"tokenize": False, "enable_thinking": False}
    prefix = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": target_text}],
                                         add_generation_prompt=False, **kwargs)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full_ids = tokenizer.encode(full, add_special_tokens=False)
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("chat template prefix differs between prompt and target; configure template explicitly")
    if len(full_ids) > max_length or len(prefix_ids) >= len(full_ids):
        raise ValueError("sample is overlong or has no assistant target tokens")
    return full_ids, [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]


def training_schedule(size, steps, accumulation, seed):
    """Legacy step-capped schedule retained for historical profiles/diagnostics."""
    if min(size, steps, accumulation) <= 0:
        raise ValueError("training schedule requires positive sizes")
    effective = max(accumulation, math.ceil(size / steps))
    order, rng = [], random.Random(seed)
    while len(order) < steps * effective:
        epoch = list(range(size))
        rng.shuffle(epoch)
        order.extend(epoch)
    return tuple(tuple(order[i:i + effective]) for i in range(0, steps * effective, effective))


def epoch_schedule(size, epochs, accumulation, seed):
    """Traverse each example once per epoch, without growing accumulation."""
    if min(size, epochs, accumulation) <= 0:
        raise ValueError("epoch schedule requires positive sizes")
    schedule, rng = [], random.Random(seed)
    for _ in range(epochs):
        order = list(range(size))
        rng.shuffle(order)
        schedule.extend(tuple(order[i:i + accumulation]) for i in range(0, size, accumulation))
    return tuple(schedule)


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
    steps: int | None = None
    epochs: int = 1
    gradient_accumulation: int = 1
    max_length: int = 2048
    max_training_tokens: int = 16384
    max_seconds: float = 1200.0
    seed: int = 42
    weight_decay: float = 0.01
    artifact_byte_cap: int = 67108864
    max_gpu_reserved_bytes: int | None = None

    def __post_init__(self):
        for name in ("rank", "alpha", "learning_rate", "epochs", "gradient_accumulation",
                     "max_length", "max_training_tokens", "max_seconds", "artifact_byte_cap"):
            if isinstance(getattr(self,name),bool) or not math.isfinite(getattr(self,name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
            if name not in ('learning_rate','max_seconds') and not isinstance(getattr(self,name),int):
                raise ValueError(f'{name} must be an integer')
        if self.steps is not None and self.steps <= 0:
            raise ValueError("legacy steps must be positive when supplied")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0 or (self.max_gpu_reserved_bytes is not None and
                (not isinstance(self.max_gpu_reserved_bytes,int) or isinstance(self.max_gpu_reserved_bytes,bool)
                 or self.max_gpu_reserved_bytes <= 0)):
            raise ValueError('invalid optimizer or GPU budget')


class HuggingFaceLearner:
    accepts_memory_context = True

    def __init__(self, directory, config, serving):
        if not config.model_revision or config.model_revision.startswith("TO_") or not config.target_modules:
            raise ValueError("freeze the actual model revision and verified LoRA target modules first")
        self.root, self.config, self.serving = Path(directory).resolve(), config, serving
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts = {}

    def train(self, examples, parent_version, memory_context=(), histories=None):
        if not self.has_artifact_capacity(3):
            raise ValueError('adapter directory limit reached; existing checkpoints require review')
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
                                           memory_context, history, example.output_kind)
            if tokens + len(ids) > cfg.max_training_tokens:
                raise ValueError("training token budget cannot cover all selected examples; memory must be retained")
            batches.append((ids, labels))
            tokens += len(ids)
        if not batches:
            raise ValueError("no valid SFT data fits the token budget")
        if cfg.steps is None and tokens * cfg.epochs > cfg.max_training_tokens:
            raise ValueError("training token budget cannot cover the requested complete passes; memory must be retained")
        allowed = {"AutoModelForCausalLM", "AutoModelForMultimodalLM", "AutoModelForImageTextToText"}
        if cfg.loader_class not in allowed:
            raise ValueError("unsupported model loader")
        if cfg.device.startswith('cuda'):
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(cfg.device)
            if cfg.max_gpu_reserved_bytes is not None:
                total = torch.cuda.get_device_properties(cfg.device).total_memory
                torch.cuda.set_per_process_memory_fraction(min(1.0, cfg.max_gpu_reserved_bytes / total), cfg.device)
        loader = getattr(transformers, cfg.loader_class)
        model = loader.from_pretrained(cfg.model_id, revision=cfg.model_revision,
                                      torch_dtype=getattr(torch, cfg.dtype)).to(cfg.device)
        if parent_version == "base":
            devices = [torch.device(cfg.device).index or 0] if cfg.device.startswith('cuda') else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(cfg.seed)
                model = get_peft_model(model, LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha,
                    target_modules=list(cfg.target_modules), bias="none", task_type="CAUSAL_LM"))
        else:
            parent = self.artifacts.get(parent_version)
            if not parent:
                raise ValueError("current adapter must be registered before continuation")
            if parent.metrics["rank"] != cfg.rank or parent.metrics["model_revision"] != cfg.model_revision:
                raise ValueError("adapter rank and base-model revision must remain fixed")
            self._validate_artifact(parent)
            model = PeftModel.from_pretrained(model, parent.path, is_trainable=True)
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not trainable or any("lora_" not in name for name, _ in trainable):
            raise ValueError("only LoRA parameters may be trainable")
        artifact_estimate = sum(p.numel() * p.element_size() for _, p in trainable) + 1048576
        if artifact_estimate > cfg.artifact_byte_cap:
            raise ValueError('adapter parameter bytes exceed the candidate disk reservation')
        before = [p.detach().cpu().clone() for _, p in trainable]
        optimizer = torch.optim.AdamW([p for _, p in trainable], lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        losses, processed_tokens, consumed, gradient_norms = [], 0, [], []
        schedule = (epoch_schedule(len(batches), cfg.epochs, cfg.gradient_accumulation, cfg.seed)
                    if cfg.steps is None else training_schedule(len(batches), cfg.steps,
                                                               cfg.gradient_accumulation, cfg.seed))
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
                                 "example_type":example.example_type, "output_kind":example.output_kind,
                                 "memory_ref":example.memory_ref,
                                 "covered_content_refs":list(example.covered_content_refs),
                                 "input_step": example.step if example.input_step is None else example.input_step,
                                 "optimizer_step": step, "tokens": len(ids),
                                 "target_tokens": sum(label != -100 for label in labels),
                                 "input_sha256": hashlib.sha256(dumps(ids).encode()).hexdigest()})
            if any(p.grad is not None for _, p in model.named_parameters() if not p.requires_grad):
                raise RuntimeError("frozen base received gradients")
            gradient_norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0, error_if_nonfinite=True)
            gradient_norms.append(float(gradient_norm))
            optimizer.step()
            if not all(torch.isfinite(p).all() for _, p in trainable):
                raise RuntimeError('non-finite adapter after optimizer update')
            losses.append(step_loss)
        if time.monotonic() - started > cfg.max_seconds:
            raise RuntimeError('training deadline exceeded; memory must be retained')
        if {row["index"] for row in consumed} != set(range(len(batches))):
            raise RuntimeError("incomplete training sample coverage; memory must be retained")
        if cfg.steps is None and len(consumed) != len(batches) * cfg.epochs:
            raise RuntimeError("incomplete requested training passes; memory must be retained")
        if not losses or not any(not torch.equal(old, p.detach().cpu()) for old, (_, p) in zip(before, trainable)):
            raise RuntimeError("no optimizer update reached the adapter")
        version = "hf_" + uuid.uuid4().hex
        path = self.root / version
        model.save_pretrained(path, safe_serialization=True)
        checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()}
        metrics = {"steps": len(losses), "losses": losses, "unique_training_tokens": tokens,
                   "schedule_mode": "epochs" if cfg.steps is None else "legacy_steps",
                   "requested_epochs": cfg.epochs if cfg.steps is None else None,
                   "planned_optimizer_steps": len(schedule),
                   "processed_training_tokens": processed_tokens,
                   "validated_examples": len(examples), "unique_consumed_examples": len({r['index'] for r in consumed}),
                   "examples_by_type":{kind:sum(e.example_type==kind for e in examples)
                                       for kind in sorted({e.example_type for e in examples})},
                   "target_tokens_by_type":{kind:sum(r['target_tokens'] for r in consumed if r['example_type']==kind)
                                            for kind in sorted({e.example_type for e in examples})},
                   "sample_coverage": 1.0, "consumed_examples": consumed,
                   "configured_gradient_accumulation": cfg.gradient_accumulation,
                   "effective_gradient_accumulation": len(schedule[0]), "sampling_seed": cfg.seed,
                   "trainable_parameters": sum(p.numel() for _, p in trainable), "base_frozen": True,
                   "model_id": cfg.model_id, "model_revision": cfg.model_revision,
                   "rank": cfg.rank, "alpha":cfg.alpha, "target_modules":list(cfg.target_modules),
                   "learning_rate":cfg.learning_rate, "gradient_norms":gradient_norms,
                   "weight_decay":cfg.weight_decay, "artifact_reserved_bytes":cfg.artifact_byte_cap,
                   "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(cfg.device) if cfg.device.startswith('cuda') else 0,
                   "cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved(cfg.device) if cfg.device.startswith('cuda') else 0,
                   "checksums": checksums, "seconds": time.monotonic()-started}
        artifact = Artifact(version, str(path), "huggingface_lora", parent_version, metrics)
        (path / "manifest.json").write_text(dumps(artifact.__dict__), encoding="utf-8")
        if sum(p.stat().st_size for p in path.iterdir()) > cfg.artifact_byte_cap:
            raise ValueError('saved candidate exceeds its configured artifact byte cap')
        self.artifacts[version] = artifact
        del optimizer, model
        if cfg.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return artifact

    def _validate_artifact(self, artifact):
        path = Path(artifact.path).resolve()
        if path.parent != self.root or path.name != artifact.version or artifact.backend != 'huggingface_lora':
            raise ValueError("adapter path is outside its configured root")
        checksums = artifact.metrics.get('checksums', {})
        if not {'adapter_config.json','adapter_model.safetensors'} <= set(checksums):
            raise ValueError('adapter lacks required file checksums')
        for name, checksum in checksums.items():
            if Path(name).name != name or hashlib.sha256((path/name).read_bytes()).hexdigest() != checksum:
                raise ValueError("adapter checksum mismatch")
        settings = json.loads((path/'adapter_config.json').read_text(encoding='utf-8'))
        if (settings.get('r') != self.config.rank or settings.get('lora_alpha') != self.config.alpha
                or set(settings.get('target_modules',())) != set(self.config.target_modules)
                or artifact.metrics.get('model_revision') != self.config.model_revision):
            raise ValueError('adapter must preserve base revision, rank, alpha and target modules')

    def prepare(self, artifact):
        self._validate_artifact(artifact)
        self.serving.load(artifact)
        self.artifacts[artifact.version] = artifact

    def has_artifact_capacity(self, limit):
        # Count even unregistered and incomplete checkpoints after a restart.
        # Preserve them for review instead of silently growing or deleting them.
        return sum(path.is_dir() for path in self.root.iterdir()) < limit

    def recover_orphans(self, versions):
        from .archive import archive_orphan
        protected = set(versions) | set(self.artifacts)
        events = []
        for path in sorted(self.root.iterdir()):
            if path.name in protected or not re.fullmatch(r'hf_[0-9a-f]{32}', path.name):
                continue
            if not path.is_dir():
                continue
            try:
                if hasattr(self.serving, 'is_registered') and self.serving.is_registered(path.name):
                    raise ValueError('orphan is still advertised remotely; preserve files for deployment-owner review')
                archived = archive_orphan(path, self.root)
                events.append({'version': path.name, 'archive': str(archived), 'state': 'archived'})
            except Exception as error:
                events.append({'version': path.name, 'state': 'recovery_failed',
                               'error': f'{type(error).__name__}: {error}'})
        return events

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

"""Real CPU gradients on a tiny action classifier; NOT a language-model experiment."""
from pathlib import Path
import hashlib
import json
import threading
import uuid

from ..contracts import Artifact, dumps


class TinyLora:
    actions = ("inspect_lock", "check_permission", "refresh_token", "renew_session", "open", "retry")
    conditions = ("vault:v1", "vault:v2", "api:v1", "api:v2")

    def __init__(self, directory, rank=8, steps=16, learning_rate=0.03, seed=42):
        import torch
        self.torch, self.root = torch, Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.rank, self.steps, self.learning_rate, self.seed = rank, steps, learning_rate, seed
        self.lock, self.models, self.paths = threading.RLock(), {}, {}
        self.models["base"] = self._new()

    def _new(self):
        torch = self.torch
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            base = torch.zeros(len(self.actions), len(self.conditions))
            base[4:] = 0.01
            a = torch.nn.Parameter(torch.randn(self.rank, len(self.conditions)) * 0.1)
            b = torch.nn.Parameter(torch.zeros(len(self.actions), self.rank))
        return base, a, b

    def _feature(self, prompt):
        state = json.loads(prompt)
        condition = f"{state['family']}:{state['version']}"
        return self.conditions.index(condition)

    def _logits(self, model, indices):
        base, a, b = model
        return (base + (2.0 * b @ a))[:, indices].T

    def act(self, observation, model_version):
        with self.lock, self.torch.no_grad():
            if model_version not in self.models:
                raise RuntimeError(f"adapter {model_version} is not loaded")
            index = self._feature(observation)
            logits = self._logits(self.models[model_version], [index])
            return self.actions[int(logits.argmax(dim=-1).item())]

    def train(self, examples, parent_version):
        torch = self.torch
        if any(e.example_type != 'decision' for e in examples):
            raise ValueError('the tiny engineering classifier supports only explicit decision_only material')
        if not examples or parent_version not in self.models:
            raise ValueError("training requires examples and a loaded parent")
        model = self._new()
        with torch.no_grad():
            model[1].copy_(self.models[parent_version][1])
            model[2].copy_(self.models[parent_version][2])
        frozen_before = model[0].clone()
        before = torch.cat([p.detach().flatten() for p in model[1:]]).clone()
        x = [self._feature(e.prompt) for e in examples]
        y = torch.tensor([self.actions.index(e.target) for e in examples], dtype=torch.long)
        optimizer = torch.optim.Adam(model[1:], lr=self.learning_rate)
        initial = float(torch.nn.functional.cross_entropy(self._logits(model, x), y).detach())
        gradient_norm = 0.0
        for _ in range(self.steps):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(self._logits(model, x), y)
            loss.backward()
            gradient_norm = max(gradient_norm, sum(float(p.grad.norm()) for p in model[1:]))
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            optimizer.step()
        final = float(torch.nn.functional.cross_entropy(self._logits(model, x), y).detach())
        changed = not torch.equal(before, torch.cat([p.detach().flatten() for p in model[1:]]))
        if not changed or not torch.equal(frozen_before, model[0]):
            raise RuntimeError("adapter did not change or frozen base changed")
        version = f"tiny_{uuid.uuid4().hex}"
        path = self.root / version
        path.mkdir()
        weights = path / "adapter.pt"
        torch.save({"a": model[1].detach(), "b": model[2].detach()}, weights)
        metrics = {"loss_before": initial, "loss_after": final, "gradient_norm": gradient_norm,
                   "steps": self.steps, "samples": len(examples), "base_frozen": True,
                   "rank": self.rank, "weight_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                   "evidence_scope": "tiny_action_classifier_engineering_only"}
        artifact = Artifact(version, str(path), "tiny_lora", parent_version, metrics)
        (path / "manifest.json").write_text(dumps(artifact.__dict__), encoding="utf-8")
        self.paths[version] = path
        return artifact

    def prepare(self, artifact):
        torch = self.torch
        path = Path(artifact.path).resolve()
        if path.parent != self.root or artifact.backend != "tiny_lora":
            raise ValueError("artifact is outside the configured backend root")
        weights = path / "adapter.pt"
        if hashlib.sha256(weights.read_bytes()).hexdigest() != artifact.metrics["weight_sha256"]:
            raise ValueError("adapter checksum mismatch")
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model = self._new()
        with torch.no_grad():
            for name, parameter in zip(("a", "b"), model[1:]):
                if state[name].shape != parameter.shape or not torch.isfinite(state[name]).all():
                    raise ValueError("invalid adapter tensors")
                parameter.copy_(state[name])
        with self.lock:
            self.models[artifact.version], self.paths[artifact.version] = model, path

    def retain(self, versions):
        with self.lock:
            for version, path in list(self.paths.items()):
                if version not in versions:
                    if path.resolve().parent != self.root:
                        raise ValueError("refusing cleanup outside adapter root")
                    self.models.pop(version, None)
                    for name in ("adapter.pt", "manifest.json"):
                        (path / name).unlink(missing_ok=True)
                    path.rmdir()
                    self.paths.pop(version, None)

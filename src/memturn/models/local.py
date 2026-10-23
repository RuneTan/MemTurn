"""Optional, process-local Transformers inference; no HTTP server or global install."""
from contextlib import nullcontext
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class LocalModelConfig:
    path: str
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    max_input_tokens: int = 6144


class LocalHFChat:
    """Frozen shared base, immutable adapter names, and one serialized GPU call."""
    def __init__(self, config, audit=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging
        logging.disable_progress_bar()
        self.torch, self.config, self.audit = torch, config, audit
        if config.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("local profile requires an available CUDA device")
            # Exercise a real kernel: device discovery alone does not prove compatibility.
            probe = torch.ones((8, 8), device=config.device)
            _ = (probe @ probe).sum().item()
        self.tokenizer = AutoTokenizer.from_pretrained(config.path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(config.path, local_files_only=True,
            dtype=getattr(torch, config.dtype), attn_implementation="sdpa").to(config.device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.lock, self.adapters = threading.RLock(), set()
        self.usage = {}

    def role(self, name, max_new_tokens):
        return LocalRole(self, name, max_new_tokens)

    def complete(self, messages, role, max_new_tokens, model_override=None):
        started = time.monotonic()
        version = model_override or "base"
        with self.lock:
            if version != "base" and version not in self.adapters:
                raise ValueError("requested local adapter has not been prepared")
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(self.config.device)
            size = inputs.input_ids.shape[-1]
            if size > self.config.max_input_tokens:
                raise ValueError("local model input budget exceeded; input was not truncated")
            if version != "base":
                self.model.set_adapter(version)
                self.model.requires_grad_(False)
            context = self.model.disable_adapter() if version == "base" and self.adapters else nullcontext()
            self.model.eval()
            with context, self.torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                    do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True)
            generated = outputs[0, size:]
            result = self.tokenizer.decode(generated, skip_special_tokens=True)
            stats = self.usage.setdefault(role, {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0})
            stats["requests"] += 1
            stats["prompt_tokens"] += size
            stats["completion_tokens"] += len(generated)
        if self.audit:
            self.audit({"event": "local_model_response", "role": role, "model_version": version,
                        "prompt_tokens": size, "completion_tokens": len(generated),
                        "seconds": time.monotonic() - started, "text": result})
        return result

    def load(self, artifact):
        from peft import PeftModel
        with self.lock:
            if artifact.version in self.adapters:
                return
            if self.adapters:
                self.model.load_adapter(artifact.path, adapter_name=artifact.version, is_trainable=False)
            else:
                self.model = PeftModel.from_pretrained(self.model, artifact.path,
                    adapter_name=artifact.version, is_trainable=False)
            self.adapters.add(artifact.version)
            self.model.requires_grad_(False)
            self.model.eval()

    def unload(self, version):
        with self.lock:
            if version not in self.adapters:
                return
            if len(self.adapters) == 1:
                self.model = self.model.unload()  # Remove adapters without merging into the base.
            else:
                self.model.delete_adapter(version)
            self.adapters.remove(version)


class LocalRole:
    def __init__(self, backend, role, max_new_tokens):
        self.backend, self.name, self.max_new_tokens = backend, role, max_new_tokens

    def complete(self, messages, model_override=None):
        return self.backend.complete(messages, self.name, self.max_new_tokens, model_override)

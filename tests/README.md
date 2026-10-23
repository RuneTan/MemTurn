# Tests

[English](README.md) | [简体中文](README.zh-CN.md)

Run core regression checks from the checkout:

```text
python -X utf8 -m unittest discover -s tests -t . -v
```

`test_core.py` covers provenance, one-entry selection, capacity budgets, access age, transactional publication, failure/restart recovery and foreground work during blocked training.

`test_backends.py` checks mocked HTTP/environment interfaces, assistant-only SFT labels, remote profile gating and real tiny LoRA gradients when PyTorch is installed.

These tests do not download weights or execute the local language model. Run `local-smoke` explicitly for that integration; measured results and remaining benchmark checks are in [validation](../docs/validation.md).

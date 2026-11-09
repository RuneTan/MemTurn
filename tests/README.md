# Test guide

[English](README.md) | [简体中文](README.zh-CN.md)

Tests use temporary databases and deterministic components to exercise memory and model updates. Run from the repository root:

```text
python -X utf8 -m unittest discover -s tests -t . -v
```

`test_material_recovery.py` focuses on retaining valid samples, duplicate references and examples, training inputs, and publication. `test_training_and_cold_selection.py` covers the complete training interface, filling cold regions within token budgets, and retaining summaries within half the actual input size. Other tests cover storage transactions, task views, retrieval, scheduling, and recovery.

See [Development](../docs/development.md) for connecting models and environments.

# Learning

[English](README.md) | [简体中文](README.zh-CN.md)

`materials.py` builds and organizes training material, `base.py` defines the learner interface, and `archive.py` provides artifact archiving and recovery tools.

Training uses `train(examples, parent_version, *, histories, memory_context=())`, supplying examples, original decision histories, and the external memories available to the job.

[Components and interfaces](../../../docs/components.md) · [Development](../../../docs/development.md)

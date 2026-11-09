# 学习

[English](README.md) | [简体中文](README.zh-CN.md)

`materials.py` 构造和整理训练材料，`base.py` 定义学习器接口，`archive.py` 提供模型产物归档与恢复工具。

训练统一调用 `train(examples, parent_version, *, histories, memory_context=())`，传入样本、原始决策历史和本次可读取的外部记忆。

[组件与接口](../../../docs/components.zh-CN.md) · [开发说明](../../../docs/development.zh-CN.md)

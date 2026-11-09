# 测试说明

[English](README.md) | [简体中文](README.zh-CN.md)

测试使用临时数据库和确定性组件，验证记忆与模型更新过程。从仓库根目录运行：

```text
python -X utf8 -m unittest discover -s tests -t . -v
```

`test_material_recovery.py` 重点检查逐样本保留、重复引用与样本、训练输入和发布。`test_training_and_cold_selection.py` 检查完整训练接口、按 token 预算填充冷区域，以及按实际输入的一半保留摘要。其余测试覆盖存储事务、任务视图、检索、调度和恢复。

接入模型和环境的步骤见[开发说明](../docs/development.zh-CN.md)。

# 测试

[English](README.md) | [简体中文](README.zh-CN.md)

`test_learning_retrieval_repairs.py` 检查样本完整覆盖、预算不足时拒绝训练、重放校验的
初始输入、适用范围、可见的工具历史、关闭记忆的评估、工具错误与环境证据的分离，
以及提示中不出现可照抄的命令占位符。

在项目根目录运行回归检查：

```text
python -X utf8 -m unittest discover -s tests -t . -v
```

`test_core.py` 覆盖来源、一条记忆选择、容量预算、访问年龄、事务发布、失败/重启恢复，以及训练等待时继续执行前台任务。

`test_backends.py` 检查模拟 HTTP/环境接口、仅助手目标计算损失、远程模板拦截，以及已有 PyTorch 时的微型 LoRA 真实梯度。

这些测试不下载权重，也不执行本地语言模型；真实模型接入请显式运行 `local-smoke`。实测结果与待完成的基准检查见 [验证状态](../docs/validation.md)。

`test_chain.py` 检查分组任务标识、Writer 预算暂停、日限额识别和原始响应审计。
核心测试还检查延后来源的重试，以及独立探针不会改变学习状态。

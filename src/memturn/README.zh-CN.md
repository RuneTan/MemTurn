# 核心模块

[English](README.md) | [简体中文](README.zh-CN.md)

`contracts.py` 定义不可变的任务、记忆、来源和模型发布记录；`config.py` 定义预算并拒绝未知字段；`tokens.py` 区分开发计数器和真实模型分词器。

| 目录 | 职责 |
| --- | --- |
| [memory](memory/README.zh-CN.md) | 状态、来源、检索与选择 |
| [tasks](tasks/README.zh-CN.md) | 环境与证据校验 |
| [models](models/README.zh-CN.md) | HTTP 与本地推理后端 |
| [learning](learning/README.zh-CN.md) | 适配器训练与文件校验 |
| [runtime](runtime/README.zh-CN.md) | 前台会话和后台发布 |

`cli.py` 提供命令；`experiments.py` 组装远程配置；`local_experiment.py` 组装本地 GPU 检查。更换这些装配入口不需要改写记忆策略。

`long_chain.py` 运行受额度约束的分组探索任务链；`chain_support.py` 提供有限重试与
无反馈的独立探针。观察者日志与 Agent 的有限记忆分离。

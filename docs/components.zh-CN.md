# 组件与接口

[English](components.md) | [简体中文](components.zh-CN.md)

| 模块 | 职责与边界 |
| --- | --- |
| `contracts.py` | 不可变的任务、经历、证据、记忆、快照与模型产物记录 |
| `config.py`、`tokens.py` | 配置检查与可注入计数器；词法计数不等于模型 token 数 |
| `memory/store.py` | SQLite 状态、来源、版本、事务、队列与恢复记录 |
| `memory/retrieval.py` | 在任务记忆视图中进行有界检索 |
| `memory/validation.py` | 来源、替换集合和训练材料准入 |
| `memory/policies.py` | 单条选择、冷区选择和半预算保留 |
| `models/agents.py` | 不绑定服务的提示式 Writer 与区域 Consolidator |
| `models/protocol.py` | 聊天客户端约定、动作格式、策略和期限控制；不含传输实现 |
| `models/budget.py` | 为注入客户端预留 token 额度并核对实际消耗 |
| `learning/materials.py`、`learning/base.py` | 有来源的问答／决策材料与学习器生命周期约定 |
| `learning/archive.py` | 带校验和的适配器归档、恢复工具 |
| `runtime/service.py` | 前后台协调、发布、退出和清理 |
| `runtime/arrival.py`、`runtime/recovery.py` | 到达负载、期限、目录锁与任务清单身份 |
| `runtime/process.py`、`runtime/resources.py` | 有界子进程通信与资源测量 |
| `tasks/base.py`、`tasks/process.py` | 环境约定与进程桥接 |
| `tasks/scienceworld.py`、`tasks/alfworld.py` | 可选环境包装；不附依赖和环境资产 |
| `matrix.py` | 对照组配置的纯转换逻辑 |

## 使用者需要提供什么

- **任务策略：** `act(observation, memories, model_version, history)` 返回环境动作或受限的记忆检索请求。`ChatPolicy` 可使用自行注入的 `ChatClient.complete(messages, model_override)` 实现。
- **Writer：** `write(episode, batch)` 返回候选记忆或 `None`。提示式实现以公开经历为证据。
- **Consolidator：** `consolidate(region, mode, batch, output_budget)` 返回候选替换集合。有界工具循环只允许查看指定区域和来源。运行系统逐条检查样本并执行整次提案的数量上限，坏样本不会连带清空好样本。
- **来源检查器：** 直接使用内置 `Validator(store)`，检查引用存在、范围、材料绑定与原始样本一致性，不需要模型、环境回放或语义判定函数。
- **学习器：** `train(examples, parent_version)` 返回 `Artifact`；`prepare(artifact)` 确保候选可用；`retain(live_versions)` 只释放无任务使用的版本。目前只有 `accepts_memory_context=True` 时才通过 `histories`、`memory_context` 关键字参数传递原生历史和许可记忆。基础双参数路径收不到历史，这是依赖历史的决策训练尚待修复的接口限制。实际优化、固定秩延续、保存和加载由使用者实现。
- **环境：** `reset(task)`、`step(action)`、`close()` 交换公开观察与 `Transition`。可选包装不会自动安装依赖。

先构造 `Store(path, config, counter)`、`Validator(store)`，再构造 `Runtime(store, writer, consolidator, validator, learner=...)`。通过 `run_task(task, environment, policy)` 执行，结束时先关闭运行系统，再关闭数据库。不预选任何后端。`learner=None` 关闭参数学习；只读运行不设置 Writer、Consolidator 和校验器。

完整方法仍需要实际工作的应用组件。接口本身不执行学习，也不能证明内化。集成责任见[开发说明](development.zh-CN.md)。

维护频率超参数为 `batch_size` 和 `capacity_interval_batches > 1`，详见[周期与恢复语义](architecture.zh-CN.md)。

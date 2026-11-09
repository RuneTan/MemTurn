# 组件与接口

[English](components.md) | [简体中文](components.zh-CN.md)

沿着一次任务到一次模型更新的过程，可以把 MemTurn 分成下面几部分。表中路径均位于 `src/memturn/`。

| 组件 | 接收什么，完成什么 | 实现入口 |
| --- | --- | --- |
| 任务策略与环境 | 策略根据观察、记忆和历史选择动作；环境执行动作并返回反馈 | `models/protocol.py`、`tasks/base.py` |
| Writer（经验记录器） | 将一次任务经历写成带来源的记忆 | `models/agents.py` |
| 记忆库与检索 | 保存经历、记忆和版本，在任务视图内检索相关条目 | `memory/store.py`、`memory/retrieval.py` |
| Consolidator（经验整理器） | 将一组记忆及其来源整理为摘要和训练材料 | `models/agents.py` |
| 来源检查与选择 | 核对引用和材料，选择学习目标及废退区域 | `memory/validation.py`、`memory/policies.py` |
| 学习器 | 用训练材料更新模型，准备和保留所需版本 | `learning/base.py`、`learning/materials.py` |
| 运行系统 | 串起任务执行、后台维护和版本发布 | `runtime/service.py` |

## 任务执行与经验整理

任务策略的 `act(observation, memories, model_version, history)` 返回环境动作或记忆检索请求。环境通过 `reset(task)`、`step(action)` 和 `close()` 提供任务交互；每步反馈用 `Transition` 表示。

Writer 的 `write(episode, batch)` 接收一次经历，返回一条候选记忆；没有值得记录的内容时返回 `None`。Consolidator 的 `consolidate(region, mode, batch, output_budget)` 接收一个区域，返回摘要集合及各自的训练材料。两者的提示式实现位于 `models/agents.py`，通过注入 `ChatClient.complete(messages, model_override)` 接入所选模型服务。

`Validator(store)` 使用库中保存的来源检查记忆和材料。[来源与训练材料](provenance.zh-CN.md)介绍引用怎样产生，以及这些检查怎样工作。

## 模型学习与发布

学习器采用统一接口：

```text
train(examples, parent_version, *, histories, memory_context=()) -> Artifact
prepare(artifact) -> None
retain(versions) -> None
```

- `examples`：所选摘要的规则问答和决策示范。
- `parent_version`：本次微调从哪个模型版本继续。
- `histories[(task_id, step)]`：对应决策之前的原始任务步骤。首步对应空元组，纯问答训练对应空字典。
- `memory_context`：本次训练可读取的外部记忆元组，可以为空。

`train` 返回模型产物 `Artifact`，`prepare` 将其准备为可用版本，随后运行系统完成发布。`retain` 根据仍在使用的版本集合清理旧产物。具体模型加载、固定秩 LoRA 训练和保存逻辑放在应用提供的学习器中。

## 把组件接起来

先构造 `Store(path, config, counter)` 和 `Validator(store)`，再将 Writer、Consolidator 和学习器传给 `Runtime(store, writer, consolidator, validator, learner=...)`。用 `run_task(task, environment, policy)` 执行任务，结束时先关闭 Runtime，再关闭 Store。

项目中的配套工具包括：

| 用途 | 模块 |
| --- | --- |
| 数据类型、参数与 token 计数 | `contracts.py`、`config.py`、`tokens.py` |
| 模型调用预算 | `models/budget.py` |
| 适配器归档与恢复 | `learning/archive.py` |
| 任务到达、期限与恢复记录 | `runtime/arrival.py`、`runtime/recovery.py` |
| 子进程通信与资源测量 | `runtime/process.py`、`runtime/resources.py` |
| 环境接入与进程桥接 | `tasks/scienceworld.py`、`tasks/alfworld.py`、`tasks/process.py` |
| 生成对照组配置 | `matrix.py` |

安装步骤、接入要求和常用参数见[开发说明](development.zh-CN.md)。

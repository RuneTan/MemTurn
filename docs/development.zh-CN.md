# 开发说明

[English](development.md) | [简体中文](development.zh-CN.md)

接入 MemTurn 的主要工作是选择任务环境和模型，并把它们接到现有的任务、经验整理和学习接口上。核心调度与存储使用 Python 3.10+ 和标准库。

## 安装与测试

在仓库根目录运行：

```text
python -m pip install -e .
python -X utf8 -m unittest discover -s tests -t . -v
```

测试使用临时数据库和确定性替身，可以离线运行，覆盖记忆检索、来源检查、训练输入、用进与废退、版本发布和恢复。需要测量进程资源时，安装可选依赖 `python -m pip install -e ".[runtime]"`。

## 接入一个应用

| 需要准备 | 接入方式 |
| --- | --- |
| 任务与环境 | 为任务分配唯一 ID，实现 `reset`、`step`、`close`；环境包装见 `tasks/` |
| 任务模型、Writer 与 Consolidator | 为提示式组件提供 `ChatClient.complete`，或实现[组件接口](components.zh-CN.md) |
| 学习器 | 实现 `train`、`prepare`、`retain`，负责固定秩 LoRA 的训练、加载和保存 |
| token 计数 | 将模型分词器交给 `TokenizerCounter`；离线组件测试可用 `OfflineCounter` |
| 运行配置 | 创建 `Config`、`Store`、`Validator` 和 `Runtime`，调用 `run_task` 执行任务 |

ScienceWorld、ALFWorld 包装需要准备对应环境依赖和资产。通过独立进程接入环境时，给 `ProcessEnvironment` 配置 `worker_command`，让工作进程响应 `configure`、`reset`、`step`、`close`。模型客户端负责实际请求、超时和取消；学习器接收每次调用传入的 `histories` 与 `memory_context`。

接入后先跑一条短任务流，检查经验写入、摘要替换、训练输入和新模型加载。再检查失败与并发情况：训练失败时保留当前模型和目标记忆，发布后新任务使用新版本，已经运行的任务继续使用原版本。部署资源上限与模型的实际 token 计数在这一步一起核对。

## 常用参数

以下是 `Config` 的开发默认值，正式实验可据任务长度和资源预算调整。

| 参数 | 默认值 | 控制什么 |
| --- | --- | --- |
| `batch_size` | 4 | 每累计多少个任务结果运行一次用进 |
| `capacity_interval_batches` | 4 | 废退间隔多少个用进批次；取大于 1 的整数 |
| `memory_token_cap` | 12000 | 活跃外部记忆的总 token 上限 |
| `region_token_cap` | 4096 | 单次交给 Consolidator 的输入区域上限 |
| `output_token_cap` | 2048 | 一次巩固输出的记忆文本上限 |
| `max_capacity_regions` | 2 | 一轮废退最多依次整理多少个区域 |
| `entry_token_cap` | 512 | 可选作学习目标的单条记忆上限 |
| `min_support_tasks` | 2 | 学习目标至少引用多少个不同来源任务 |
| `max_examples_per_job` | 32 | 一次整理输出中最多保留多少个训练样本 |

默认用进／废退周期为 4／16 个任务结果，成功、失败和拒绝均计数。同轮到期时先用进，再废退；后台繁忙时合并逾期周期，处理当前到期工作。只读评估使用独立的读取路径，维护计数与记忆热度保持原值。

**一个区域是一组记忆，2 个区域并不等于 2 条记忆。** 废退按冷度扫描候选，在 `region_token_cap` 内持续装入完整条目。每个区域的保留预算是 `min(floor(实际输入 token 数 / 2), output_token_cap)`；`return_token_ratio` 在当前设计中固定为 `0.5`。

更多存储、队列、检索和版本预算见 [`config.py`](../src/memturn/config.py)。底层模型计数由所选分词器决定。

## 做对照与查看记录

[`matrix.py`](../src/memturn/matrix.py) 可生成仅整理外部记忆、随机选择学习目标、训练后保留记忆、只用一种训练材料等对照配置。应用在相同任务流与资源预算下运行这些配置，记录任务收益和开销，用来比较各组件的贡献。

为 Runtime 传入 `audit` 回调可以收集写入、巩固、训练材料筛选、训练与发布事件。抽查材料时可从 `qa_text_overlap` 等提示定位样本；工程回归测试的入口与分类见[测试说明](../tests/README.zh-CN.md)。

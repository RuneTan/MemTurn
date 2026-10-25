# MemTurn 集群交接与工程补齐

日期：2026-10-24。用途：交给合作者审阅、安装和验收，再共同选择模型与正式实验方案。**这是研究工程交付，不是已经取得标准任务收益或集群服务指标的报告。**

## 保持的研究基线

默认仍是一个用户、一条任务流、一个有界外部库、冻结基座和固定秩 LoRA。Writer 只写当前经历；同一个 Consolidator 生成摘要与关联训练材料。常规批次最多选择一条合格巩固记忆，支持任务数优先；默认混合材料为一条规则问答和真实来源决策。学生屏蔽所教记忆及已识别同源旁路，完整遍历合格样本；成功发布后仅退出对应版本。容量维护仍是冷区域、实际区域半预算、动态 k，且不触发该次训练。

没有增加独立教师、强化学习、动态增长调度或专项抗遗忘模块。新增的随机选择、QA-only、保留记忆等只作为显式实验开关，不改变默认基线。

## 本次已接入的工程工作

| 工作 | 实际行为 | 验证边界 |
| --- | --- | --- |
| 任务流恢复 | 冻结清单哈希、连续完成水位和有界乱序窗口；结果、收益、待写来源与流完成状态共同提交 | 窗口外旧任务不重新计数；中断的模拟器任务从 reset 重试，不承诺外部世界动作 exactly-once |
| 运行目录互斥 | 操作系统文件锁，进程退出自动释放；相同运行只能有一个远程协调进程 | 不以 PID 文件年龄推断锁可删除 |
| 进程时限 | HTTP、原生环境、领域校验和 SFT 使用有界 JSON 子进程；超时终止任务进程树 | Linux 用进程组，Windows／嵌套进程配合 psutil；服务器收到远程请求后的计算仍由服务器调度 |
| 有界到达服务 | 固定到达间隔、并发与排队上限；排队时间计入期限；拒绝和超时也记结果与效用 | 恢复后另开一个服务测量分段，不把停机时间当作已服务时间；报告直方图 p95 上界而非伪精确分位数 |
| 计算预算 | 用冻结 tokenizer 的真实聊天模板预留输入加最大输出；服务端 token 账目不一致则拒绝结果；预算持久化 | 未知失败或崩溃保留全部预留，分别报告保守计费和已确认消耗，不把缓存 token 当作免费计算 |
| 训练预算 | 持久化训练 tokens／更新次数额度，联合准入；作业完成后按实耗结算 | 完整样本遍历仍是发布前提；额度不足不截短样本或偷偷减少遍历 |
| 内存和磁盘 | 训练载荷／张量预留、训练进程树 RSS 监测、CUDA 分配器限制与峰值、主库/WAL/SHM 预留、候选文件上限 | RSS 是采样总和，会重复计共享页并可能漏短峰值；真正的全进程硬内存限制和服务器显存额度仍须集群配额 |
| 模型与协议 | 冻结部署说明文件的 SHA-256，核对实际服务模型名称；不可变 LoRA 名称、共享路径核对、加载后动作协议探测 | 部署说明是操作者证明，不是远程权重的密码学证明；协议探测不等于知识掌握验证 |
| 原生经验判定 | 提供 `observed_transition` 的保守判定与完整环境回放，并能在独立进程运行自定义领域判定 | 只认证确实发生且有原生分数进展的片段，不认证自由文本一般化或最优动作 |
| 对照驱动 | 生成、运行、汇总显式对照配置；校验共享预算和任务流 | 相同上限不等于相同实际训练量；汇总同时展示实耗，不自动产生显著性或机制结论 |
| 孤立候选恢复 | 完整归档并校验后清理；当前和在途版本保留；归档可手动恢复 | 远端仍登记的未知候选只读检查、保留文件，交部署负责人核对，不自动卸载 |

## 首次接手：先做不需要模型的检查

在解压后的 `MemTurn` 目录运行，建议独立 Python 3.10+ 环境：

```bash
python -X utf8 scripts/cluster_inventory.py --output cluster_inventory.json
python -X utf8 run_memturn.py doctor
python -X utf8 -m unittest discover -s tests -t . -v
python -X utf8 run_memturn.py demo --config configs/smoke.json --output runs/first_offline_check
```

库存脚本只读取硬件、软件版本、磁盘、已配置的调度资源和 GPU 信息；不会打印密钥，也不会下载模型。测试中的 HTTP 服务只监听本机回环地址。缺少 PyTorch／训练依赖／psutil 时，对应可选测试会跳过；请保留完整日志，不把跳过项写成通过。

远程运行需安装项目的 `runtime` 依赖，包括 psutil；训练依赖在确定集群 CUDA／驱动后安装。示例：

```bash
python -m pip install -e '.[runtime]'
# 按集群验证的 CUDA/PyTorch 方案安装训练依赖后：
python -m pip install -e '.[train]'
python -m pip freeze > cluster-resolved-requirements.txt
```

`configs/local-windows.lock.txt` 是已有笔记本环境的历史记录，不能直接充当 Linux 集群锁文件。原生环境可以用独立 Python 环境；在配置中的 `environment.worker_command` 指定该解释器及 `scripts/native_worker.py`。环境包版本会在 worker 中核对，任务清单固定具体实例、版本、种子与步数。环境资源和模型权重不在交付包内。

## 配置与运行

`configs/scienceworld.example.json`、`configs/alfworld.example.json` 保留 `TO_*`。正式模型、设备、额度、负载和效用值尚未决定；数值型开发默认值也不表示已优化。复制模板后填写：

1. 三个角色的模型服务名称、不可变模型／tokenizer revision、上下文上限、各角色总 token 额度。
2. 各服务的部署说明文件和文件 SHA-256，参考 `configs/deployment.example.json`；记录无密钥的启动参数和服务软件版本。
3. 任务模型的加载类、实际支持的 LoRA 目标层、共享 adapter 路径。训练继续沿用同一基座 revision、rank、alpha 和目标层。
4. 原生任务清单、包版本、进程时限及经过验证的领域校验器。
5. 到达间隔、任务期限、并发、排队容量、按任务族的收益归一化与失败／拒绝／超时值。
6. 实际 RAM／GPU／存储配额，训练 tokens 与更新次数的总额度。每个运行使用独立目录。

模型名称和地址可使用独立服务；密钥只通过 `api_key_env` 指定的环境变量注入。共享目录必须同时被训练进程和推理服务按配置路径访问。

```bash
python -X utf8 run_memturn.py check-profile --profile configs/verified_profile.json
python -X utf8 run_memturn.py run --profile configs/verified_profile.json --output runs/experiment_a
python -X utf8 run_memturn.py run --profile configs/verified_profile.json --output runs/experiment_a --resume
```

配置缺项时检查返回退出码 2；检查完整仅表示静态配置完整，不表示远程验收成功。`--resume` 要求相同配置与任务清单，不能根据旧任务计数猜测新流的位置。运行目录和文件系统需支持可靠文件锁及 SQLite；不要让多个机器写同一个状态库。调度模板见 `scripts/slurm_run.template.sh`，资源字段须先与集群负责人确认。

`arrival_policy=periodic` 才走到达／排队驱动。`wait_for_background` 保留为逐任务等待的开发模式，不能据此宣称不停机服务。队列加并发必须小于恢复窗口；早期任务尚未完成时，窗口会阻止控制记录无限增长，已经错过期限的后续到达仍记为超时。

## 原生领域校验器的边界

内置窄范围工厂为 `memturn.tasks.observed:make_verifier`，其 `verification_kwargs` 需包含 `environment` 配置。经验正文是明确标记为 `observed_transition` 的结构化记录，包含原观察摘录、原动作与反馈摘录；适用条件使用指定格式。Writer／Consolidator 自动获得相应格式说明。

认证需要来源一致、动作确实可用、原生分数在该步增长、完整回放一致。多个来源必须支持同一片段。这个验证器证明“该转移在这些来源中发生过”，不证明“所有满足片段条件的未来状态都应执行此动作”。它也不会为了提高小模型成功率改写原生观察或加入标准答案。

如果正式实验需要更广的一般性经验，应另实现领域谓词，继续通过同一个进程隔离和来源核验接口；缺少可靠判定时拒写／不训练。选择此内置范围或扩大范围，留待与合作者讨论。首轮官方原生效果实验尚未运行。

## 对照配置与实际预算

```bash
python -X utf8 scripts/build_experiment_matrix.py --profile configs/verified_profile.json --output configs/review_matrix
# 完成每个实验独立的共享 adapter 路径等字段后，再显式指定要执行的组：
python -X utf8 scripts/run_experiment_matrix.py --matrix configs/review_matrix --output runs/matrix --arms baseline random_selection
python -X utf8 scripts/compare_runs.py runs/matrix/baseline runs/matrix/random_selection --output comparison.json
```

目前支持十组：基线、合格池随机选择、仅外部巩固、训练后不退出、仅 QA、仅决策、简单冷项淘汰、原始决策 SFT、未巩固外部记忆、无记忆。

“原始决策 SFT”准确指：从已经通过 Writer 来源校验的未巩固条目中选择，直接使用其真实来源观察／动作，门槛显式改为一个来源任务；它不等于把所有原始轨迹无筛选投入训练。QA-only、决策-only 会改变可训练材料与候选资格；这些是系统消融。支持数与随机选择使用相同合格门槛，才是选择规则对照。

配置生成不执行实验；执行器只运行显式指定的组。共享预算指纹包括模型、环境、负载、效用和预算。输出仍须检查实际训练 token、更新次数、任务完成数及失败率；如果不一致，不能写成“相同实际训练量下的纯选择效应”。所有测试种子、流长度、变化方案和正式对照组合仍需冻结。

## 故障、资源和恢复

训练被终止时不发布候选，已提交摘要保留，所选外部记忆不退出。未知调用保留计算额度，避免重启后把可能已经付出的计算当作免费。临时训练载荷走有界管道，不形成可被下一轮偷偷回查的训练历史。

SQLite 禁止事务中反复缓存溢写，并在事务边界清空 WAL；预留三倍主库上限覆盖主库、WAL 和 SHM。训练前预留候选 adapter，核验实际文件大小；训练进程监测 RSS，记录 CUDA 分配器峰值。软件监测允许在采样间隔内短暂超出阈值，GPU 分配器额度也不等于所有 CUDA／推理服务器占用。应结合 Slurm/cgroup 内存限制、作业时限、磁盘配额，以及推理服务显存／CPU adapter 缓存限制完成硬预算验收。

HTTP 成功加载／卸载和 `/models` 的变化只能证明对应接口状态，不能单独证明显存缓存立即释放。按所选服务版本核验实际缓存行为，并限制 `max_loras`、`max_cpu_loras` 等资源。动态 LoRA API 的协议依据见 [vLLM 官方文档](https://docs.vllm.ai/en/latest/features/lora/)；这些管理接口应部署在受控的研究服务中。

孤立文件的归档与手动恢复见 [专门说明](adapter_archive_recovery.zh-CN.md)。归档、审计日志和分析报告不作为学习状态回查，但仍消耗真实磁盘；恢复后重新参与运行的文件必须重新计入预算。由于不能从本地目录独立判断远端是否仍有用户，远端仍登记的未知版本会保留，供部署负责人核对。

## Auto-Dreamer 复用核查

截至本次核查，在 [论文主页](https://arxiv.org/abs/2605.20616)及[原文的 Artifact Details](https://arxiv.org/html/2605.20616v1#A3)中未确认到作者直接提供的可下载训练后 Consolidator 权重与配套实现入口；原文列出了 OpenTinker 和 verl 依赖。这个结果是“当前没有核实可复用产物”，不等于证明产物不存在。交付代码继续明确标注提示式区域重写原型，不冒称 Auto-Dreamer 完整复现，也不将它的模型选择自动变成 MemTurn 正式模型。

## 交付验收与尚待外部条件

本次验证日志、代码校验清单、文件清单及两轮隔离 SFT 记录放在交付包 `validation/` 中。其具体测试计数以 `validation/summary.json` 为准。新增检查包括长期去重、乱序恢复、提交回滚、目录锁、子进程／子孙进程终止、输出及内存限制、超载计费、真实本机 HTTP 调用与恢复、部署证明不匹配、未知远端候选保护及消融行为。

尚待你们提供集群信息后完成的是：所选模型与 tokenizer／加载器／目标层的真实兼容性，服务端权重与启动记录核对，原生任务实例及领域校验覆盖率，集群总内存与显存峰值、并发竞争和延迟目标，正式任务流及预算冻结，以及长程结果和统计分析。多机分布式训练不在当前固定单学习作业实现内；如果选定模型需要它，应单独扩展并验收，不能将此包当作已支持分布式 SFT。

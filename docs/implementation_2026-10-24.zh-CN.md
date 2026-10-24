# 方案落实与验证状态：2026-10-24

[English](implementation_2026-10-24.md) | [简体中文](implementation_2026-10-24.zh-CN.md)

已把约定的混合监督候选方案接入系统。训练执行成功不能证明经验已经内化。

## 已落实的训练流程

常规 Consolidator 在同一次操作中输出正文、贡献父项、来源、规则问答和原生决策示范，不再增加教师或改写 Agent。当前使用提示式实现，尚不是 Auto-Dreamer 训练后的检查点。

每条样本明确标记类型、输出类型、所属记忆版本、正文覆盖范围与来源。输出的 `content` 或 `span:start:end` 引用会绑定到新记忆版本；决策来源使用 `task_id` 与从零开始的 `step`。

规则问答不在输入中展示所教正文，目标从正文或指定片段忠实复制，输出为普通文本。决策示范保留来源当时的公开观察、此前环境历史和实际动作，输出为动作 JSON。不再默认把后续动作移到先前状态，也不为验证模型改造环境答案。

默认 `training_material: mixed` 要求两种样本同时有效。每次巩固的 `max_examples_per_job: 32` 预算包含规则问答；问答不会增加独立任务支持数。经验本身有效但训练材料不合格时，保留外部经验、移除训练材料，不自动退回动作示范训练。经验的来源依据不合格时，经验本身也不能入库或替换旧区域。微型分类器及历史动作诊断显式使用 `decision_only`。

仍按不同任务支持数选至多一条记忆。学生屏蔽该记忆、父项和可识别同源项，预先检索允许读取的其他经验。训练保留原生环境历史，不继承旧检索正文或查询，避免绕过屏蔽。静态 SFT 内部不调用新工具。

样本打乱后完整训练一遍，梯度累积为一，继续更新同一固定秩 LoRA。候选适配器准备完成后发布并退出选中版本。当前直接退出策略没有额外掌握度门槛，本轮没有引入新门槛。

## 检索与记忆增长统计

轨迹单独保存初始检索和追加检索，Writer 可以看到任务 Agent 实际读过的经验。原生任务的私人清单、数据集条件标签不会发送给经验组件。

同一任务历次有上限的检索记录留在上下文，重复正文只出现一次；不恢复固定记忆区域。来源步骤编号只计算环境动作。

`stats.growth` 分别累计初始记忆量、录入量、常规巩固净释放量、内化退出量和容量维护净释放量，满足：

```text
当前正文 token = 初始正文 token + 累计录入
                 − 常规巩固净释放 − 内化退出 − 容量维护净释放
```

常规巩固增加正文时，净释放量可以为负。重复录入不重复计数；事务失败连同计数一起回滚。这里只保存固定数量的累计值；旧数据库以当前存量作为起点，不伪造历史。

这些统计支持后续分析增长速度，尚未启用增长率自适应调度器。常规巩固仍按固定批次触发；冷区域、半数 token 返还预算和容量维护不训练的规则保持一致。

## 原生环境与模块隔离

Windows 模型环境和 Linux 模拟器环境通过逐行 JSON 的独立进程通信。`ProcessEnvironment` 与 `native_worker.py` 只传递原生任务、观察、动作、反馈和奖励，不承担训练或记忆策略。没有修改全局 Python 或 Java 安装。

`install_native_linux.py` 在项目中创建 `.venv-native-linux`，固定 ScienceWorld 1.2.3、ALFWorld 0.4.2 和 PyYAML 6.0.3，保存实际依赖清单。便携 Java 17 从 [Adoptium 官方接口](https://api.adoptium.net/)下载，核验 SHA-256 后保存在 `.cache/native`，并记录来源。

`prepare_native_checks.py` 下载 ALFWorld 官方文本资源，记录来源和校验和，生成明确指定训练集实例的接口检查清单，不向模型暴露专家计划。ScienceWorld 不启用简化规则。安装依赖、资源、权重和原始产物不进入 Git。

Linux／WSL 中运行：

```text
python3 scripts/install_native_linux.py
.venv-native-linux/bin/python scripts/prepare_native_checks.py --kind scienceworld --wsl-distro 你的发行版名称
.venv-native-linux/bin/python scripts/prepare_native_checks.py --kind alfworld --wsl-distro 你的发行版名称
```

准备脚本生成供 Windows 调用的进程命令，随后在 Windows 项目目录运行：

```text
.venv-local/Scripts/python.exe scripts/check_native.py --config runs/native_setup_20261024/scienceworld.check.json --output runs/新的ScienceWorld检查目录
.venv-local/Scripts/python.exe scripts/check_native.py --config runs/native_setup_20261024/alfworld.check.json --output runs/新的ALFWorld检查目录
```

直接使用 Linux 服务器时，将 `environment.worker_command` 改为隔离 Python 和 `scripts/native_worker.py` 的绝对路径。接口检查不调用模型、不训练，清单不能当作正式基准运行。

`NativeReplayVerifier` 可重放核查原始观察、动作反馈、证据字段、奖励和成功状态。动作合法不等于动作优质；来源重放也不能证明任意自然语言归纳成立。因此，经验支持仍需要独立领域判定器，缺少时拒绝认证。`verification_factory` 允许替换这部分，而不修改记忆策略。

## 实测与边界

ScienceWorld 和 ALFWorld 均完成两个原生训练集实例的启动／动作检查，无模型调用、无训练。记录分别在 `runs/native_scienceworld_interface_20261024/` 和 `runs/native_alfworld_interface_20261024/`；这些不是任务求解结果。另独立重放了两个 ScienceWorld 来源和一个 ALFWorld 来源，均匹配；未提供领域判定时，摘要支持均保持拒绝认证。此 WSL 挂载目录上的 ALFWorld 单次检查耗时约 55–105 秒，主要涉及初始化，不代表在线服务表现。

回归测试 **62 项通过**，新增覆盖训练目标绑定与忠实性、不自动降级、隐私字段隔离、重放防篡改、进程重启、增长统计守恒及回滚。

本地 RTX 5080 Laptop 上的 Qwen3-1.7B 完成两轮连续 rank-8 LoRA 训练。每轮完整使用 1 条诊断问答和 2 条未修改的 ScienceWorld 决策记录，共更新 3 次，覆盖率均为 100%，基座保持冻结。候选加载及继续训练成功，记录在 `runs/mixed_sft_interface_20261024/report.json`。

这条问答是明确标识的接口材料，没有经过领域校验，也不是 Consolidator 的有效研究产出；此次没有退出外部记忆。训练后的问答探针仍输出环境动作对象。因此，可以确认混合目标进入了训练、适配器可以继续更新，但不能确认模型已学会可靠的规则文本回答，更不能据此声称原生长期学习已经有效。

仍需完成原生领域证据判定、真实经验组件质量检查、任务求解与长链对照；远程硬件和 Qwen3.5-9B 服务／LoRA 兼容性待验证。全程在线需要进一步测量推理与训练争用。本轮没有用额外教师、自适应调度器、RL 或掌握度门槛代替这些未决事项。

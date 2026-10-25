# 本地 GPU 接入

[English](local_model.md) | [简体中文](local_model.zh-CN.md)

检查日期：2026-10-23。以下是开发环境接入记录，不是标准任务成绩，也不能证明记忆内化提高了任务表现。

## 硬件与环境隔离

本地设备为 RTX 5080 Laptop GPU，显存 16,303 MiB。原系统 PyTorch CUDA 构建缺少 `sm_120` 内核。项目另建 `.venv-local`，使用 PyTorch 2.10.0+cu128，已通过真实 CUDA 矩阵运算并确认支持 `sm_120`，系统 Python 保持不变。

模型为 [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)，固定版本 `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`，以 BF16 加载。它是较大远程模型的开发替代，不是 Auto-Dreamer 复现。PyTorch 安装包使用[官方 CUDA 12.8 来源](https://pytorch.org/get-started/previous-versions/)。

| 项目 | 位置或版本 |
| --- | --- |
| Python 环境 | `.venv-local/`，Python 3.10.6 |
| 权重与 tokenizer | `models/qwen3-1.7b/` |
| 本地配置 | `configs/local_qwen3_1_7b.json` |
| Transformers | 5.2.0 |
| PEFT | 0.19.0 |
| Accelerate | 1.12.0 |
| 完整依赖快照 | `configs/local-windows.lock.txt` |
| 模型后端 | `src/memturn/models/local.py` |

只有选定后端才导入这些依赖，核心样例仍可只用标准库运行。运行时加载本地文件，下载是另一步显式操作，并固定公开模型版本。

## 复现

在 Windows 项目目录中运行：

```powershell
.\scripts\install_local.ps1
.\.venv-local\Scripts\python.exe -X utf8 -m unittest discover -s tests -t . -v
.\.venv-local\Scripts\python.exe -X utf8 run_memturn.py local-smoke --tasks 8 --operators fixture --train
```

确定性开发组件产生可核查的记忆，任务动作仍来自真实语言模型，训练使用实际 Hugging Face SFT 后端。如需明确检查模型生成的 Writer／Consolidator 输出：

```powershell
.\.venv-local\Scripts\python.exe -X utf8 run_memturn.py local-smoke --tasks 8 --train
```

每次运行新建被 Git 忽略的 `runs/local_*` 目录。`report.json` 记录任务结果、模型用量、GPU 峰值、依赖版本和事件数量；`audit.jsonl` 包含生成回复、校验拒绝和 adapter 发布记录。这些资料不能通过记忆检索读取。

## 实测记录

| 配置 | 任务成功 | adapter 发布 | CUDA 分配峰值 | 观察 |
| --- | --- | --- | --- | --- |
| 本地任务模型，确定性 Writer／Consolidator，开启 SFT | 5/8 | 2 | 10.70 GiB | 完成真实 adapter 延续及发布，无后台错误 |
| 任务／Writer／Consolidator 全部使用本地模型，开启 SFT | 8/8 | 0 | 4.06 GiB | 八个 Writer 候选均未通过证据校验，没有记忆入库 |
| 本地任务模型，Groq GPT-OSS 120B Writer，确定性 Consolidator，开启 SFT | 4/4 | 1 | 9.09 GiB | 四条 Writer 记忆入库，一次巩固及发布，无后台错误 |

前两次记录为 `local_20261023T145345681297Z` 和 `local_20261023T145602611706Z`，完整目录、权重及原始运行数据未随仓库发布。这是组件和状态轨迹不同的短程接入检查，不是匹配条件的对照；成功率差异不能解释为训练效果。

后续 Groq 试跑为 `local_20261023T151237196728Z`，其 adapter 在四任务流末尾才发布，未用于后续任务。见 [Writer 接口说明](writer_api.zh-CN.md)。

这些历史 SFT 作业采用 rank 8、`q_proj`／`v_proj`、1,605,632 个可训练参数和两次优化器更新。第二个发布 adapter 从第一个延续。已检查有限损失和 LoRA 参数变化，但这不证明学习质量。第二版在任务流末尾发布，也不能据此判断其后续表现。

## Writer 的限制与拒绝机制

初始模型 Writer 会混淆反馈与动作目标、来源索引和适用标签。来源示例现支持任务／步骤引用，由程序复制原观察和动作，再独立校验。这样模型不必逐字重写很长的来源文本，但并不因此证明其结论正确。

本地 1.7B Writer 仍会生成无支持的摘要或错误条件，这些候选被拒绝。前台继续执行，没有有效入库记忆就没有后续巩固或 SFT 材料。目前暂缓进一步调优小模型 Writer；确定性组件继续用于工程检查，较大的远程组件仍需验证。尚未做更强模型的对照，不能区分模型容量和提示的因果作用。

## 记忆容量是暂定工程设置

本地配置的活跃记忆上限为 **8000 个任务 tokenizer tokens**，统计活跃条目的适用条件和正文之和。这是接入起点，不是优化结果，也不是由显存换算而来。记忆库在外部，每任务最多接收 512 个检索 tokens，单次巩固区域最多 2400 tokens，候选输出最多 1200 tokens；容量返还还受实际处理输入一半的预算限制。

通用默认值为 12000 tokens。原离线样例使用 1200 个近似词项单位，不能与 Qwen tokens 直接比较。来源与训练样本另有 32 MiB 上限，元数据、快照和数据库页分别核算。

八任务运行未触及记忆上限，容量行为由回归测试覆盖。正式实验应测量无约束增长，并选择足以触及预算的任务流。2k／8k／32k tokens 可以作为各方法共享的敏感性网格，但它只是实验建议，尚未验证。

## 待检查事项

本地模型串行处理生成和 adapter 切换，短程运行在任务间等待维护，不能测量不停机服务或 GPU 竞争。真实标准任务资源、领域证据判定、原训练后 Consolidator 权重、远程 LoRA 兼容性及匹配预算的长程比较仍待完成。

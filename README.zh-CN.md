# MemTurn

[English](README.md) | [简体中文](README.zh-CN.md)

**资源约束下持续学习 Agent 的记忆内化**

MemTurn 研究如何让 Agent 面对持续变化的任务流，在有限外部记忆、固定秩 LoRA、明确学习成本与服务预算下，持续从自身经验中学习。

**当前状态：可执行研究原型。** 记忆生命周期测试与微型梯度训练已在本机运行；独立的 Qwen3-1.7B 后端用于本地 GPU 接入检查。ScienceWorld、ALFWorld、原版训练后的 Auto-Dreamer Consolidator 和长程对照实验仍待完成。工程检查尚不能证明研究优势。

## 快速开始

核心小环境只需要 Python 3.10 及以上，使用标准库：

```text
python run_memturn.py demo --config configs/smoke.json
python -X utf8 -m unittest discover -s tests -t . -v
```

已有 PyTorch 时，可在演示命令末尾加 `--train-tiny`，验证微型动作分类器的 CPU 梯度更新。它与语言模型 SFT 是两个独立后端。

## 本地 GPU 模型

本地配置使用固定版本的 [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) BF16 权重。Windows 脚本将 CUDA 12.8 版 PyTorch 安装到 `.venv-local`，模型下载到 `models/`：

```powershell
.\scripts\install_local.ps1
.\.venv-local\Scripts\python.exe run_memturn.py local-smoke --tasks 8
.\.venv-local\Scripts\python.exe run_memturn.py local-smoke --tasks 8 --operators fixture --train
```

第一种运行由模型承担任务、Writer 和 Consolidator；第二种使用可校验的小环境经验组件，单独验证真实任务模型推理和语言模型 LoRA SFT。两者都使用开发小环境，默认每个任务后等待后台，不用于测量不停机服务。

硬件、实测结果和复现步骤见 [本地模型说明](docs/local_model.md)。

可通过 `--writer-profile` 单独选用固定的 API Writer。免费候选与配置方法见
[接口说明](docs/writer_api.md)。Groq 小规模试跑已接纳四条记忆，并完成一次本地
LoRA 发布；Gemini 尚未验证。

## 系统设计

每个任务持有固定模型版本与记忆视图，开始时检索一次，并可按需追加有限次检索。后台负责写入有来源的经验、巩固有限区域，并先提交通过校验的替换记忆。

独立选择器每个常规批次至多选一条合格的巩固记忆，优先选择获得更多不同任务实例有效支持的结果。学生不能检索选中记忆、其父项和已识别的同源记忆。训练继续更新固定秩 LoRA；候选版本准备完成后，才更新发布指针并移除选中记忆的对应版本。

容量维护单独处理冷记忆：在区域预算内选取最久未访问记忆的初始比例，巩固后按有效任务支持数排序，将完整结果装入实际输入 token 数一半的返还预算。返还条数自然产生，这一路不训练。拒绝写入的事件会被记录。

## 目录说明

| 位置 | 职责 |
| --- | --- |
| [docs](docs/README.zh-CN.md) | 架构、部署、验证与文献 |
| [configs](configs/README.zh-CN.md) | 离线、本地和远程配置 |
| [scripts](scripts/README.zh-CN.md) | 隔离安装与固定版本下载 |
| [src](src/README.zh-CN.md) | 记忆、模型、环境与训练模块 |
| [tests](tests/README.zh-CN.md) | 生命周期、失败、并发与后端检查 |

建议从 [架构](docs/architecture.md)、[部署](docs/setup.md) 和 [验证状态](docs/validation.md) 开始。主要目录均提供独立的中英文 README；详细技术文档目前以英文为主。

## 基准实验阶段

首轮范围是 **ScienceWorld 与 ALFWorld**，WebArena 暂缓。远程配置按 Auto-Dreamer 选择任务模型 `Qwen/Qwen3.5-9B`，Writer 和 Consolidator 基座 `Qwen/Qwen3-14B`。当前提示式 Consolidator 尚不是原论文训练后的模型。

正式实验前，需要验证远程硬件、模型版本、LoRA 目标层、环境资源、任务清单和基准证据校验器：

```text
python run_memturn.py check-profile --profile configs/scienceworld.example.json
```

模板缺项时会在模型执行前停止。任务最终成功，不能自动证明所有中间动作都适合成为监督数据。

## 复现与发布范围

报告、审计、状态库和适配器写入 `runs/`。权重、虚拟环境、缓存、凭据和生成产物不进入 Git；通过固定配置与依赖记录复现安装。

仓库只发布 `MemTurn` 项目，父工作区的研究笔记、文献 PDF 与凭据保持独立。克隆方式：

```text
git clone https://github.com/RuneTan/MemTurn-Memory-Internalization-for-Continual-Agents-under-Resource-Constraints.git MemTurn
cd MemTurn
```

基准表现、长程稳定性和创新性仍需实证检验。来源见 [原始文献](docs/literature.md)。项目软件许可证尚未选定。

# 开发与远程部署

[English](setup.md) | [简体中文](setup.zh-CN.md)

## 本地检查

纯记忆开发样例只依赖标准库，Python 3.10+ 即可。在项目目录运行：

```text
python run_memturn.py doctor
python run_memturn.py demo --config configs/smoke.json
python -m unittest discover -s tests -t . -v
```

若已有兼容的 PyTorch，可以给 demo 增加 `--train-tiny`；这不会下载 Qwen。输出进入新建且被 Git 忽略的 `runs/` 子目录，这些命令不会修改系统 Python 安装。

如需安装项目，先创建独立环境，再运行 `pip install -e .`。`train`、`scienceworld` 和 `alfworld` 是可选依赖范围，不是已经验证的远程锁文件。

## 首轮标准任务接入

笔记本上的独立 Qwen3-1.7B 推理和语言模型 SFT 见[本地模型说明与实测限制](local_model.zh-CN.md)。该路径不需要标准任务资源或远程服务器。

首轮环境为 ScienceWorld、ALFWorld，WebArena 暂缓。正式任务模型、Writer 和 Consolidator 仍待与合作者讨论，远程模板有意保留模型占位。

Auto-Dreamer 的 Consolidator 从 Qwen3-14B 训练而来。直接使用基础 Qwen3-14B 加提示，只能称为开发变体，不是原训练后权重。见 [Auto-Dreamer 第 5.1 节](https://arxiv.org/html/2605.20616v1)。

先运行：

```text
python run_memturn.py check-profile --profile configs/scienceworld.example.json
```

该命令报告未完成字段，不连接服务、不下载权重，也不打印密钥。确认以下内容后再填写配置副本：

1. GPU 型号、数量、显存，Linux／Python 版本，存储路径和计算预算。
2. 各角色服务地址和固定的模型／tokenizer 版本。使用 `.env.example` 中的变量名，密钥留在仓库外。
3. 实际 Consolidator 权重，或明确标注的提示式变体。
4. 验证过的 LoRA 目标层和任务模型加载类。Qwen3.5 属于多模态架构，不能直接假定 `AutoModelForCausalLM` 可用。
5. 训练和推理都能访问同一有界 adapter 目录。vLLM 发布使用不可变名称及动态 LoRA 接口；所选模型和加载器组合必须真实支持 LoRA，该远程兼容性尚未在本地验证。参照 [vLLM 文档](https://docs.vllm.ai/en/latest/features/lora/)。
6. 环境资源和固定 JSONL 任务清单，包含唯一 ID、任务族、版本、种子、步数上限及环境所需信息。
7. 对应任务域的 `EvidenceVerifier`，核查来源支持和监督目标。`verification.py` 会拒绝未经核查的主张；模拟器最终得分良好，不代表每个中间动作都是合格监督。

ScienceWorld 需要 Java 和环境包，并需固定包版本；[官方说明](https://github.com/allenai/ScienceWorld)记录了可能影响轨迹的变更。ALFWorld 需要文本环境、游戏资源和 YAML 配置，见[官方安装说明](https://github.com/alfworld/alfworld)。

两种环境已安装在独立的项目 Linux 环境中，见[实现记录](implementation_2026-10-24.zh-CN.md)。这并未完成上述领域校验及远程部署验收。

检查完成后的实际运行入口为：

```text
python run_memturn.py run --profile configs/your_verified_profile.json --output runs/your_new_run
```

示例配置含未决字段，不能原样启动正式实验。远程入口现已支持有界周期到达、显式 `--resume`、不可变任务清单和有限乱序完成窗口。已有窄范围转移片段校验，自由文本语义主张仍须可靠判定。进程时限、预算、部署说明核对、对照工具及集群待验收项见[集群交接](cluster_handoff.zh-CN.md)。离线样例不能证明分布式 SFT 或实际服务水平已经完成。

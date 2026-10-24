# 安装脚本

[English](README.md) | [简体中文](README.zh-CN.md)

`install_local.ps1` 创建 `.venv-local`，安装固定版本的 Windows CUDA 依赖并下载本地模型。脚本检查命令是否失败，不修改全局 Python。

`download_local_model.py` 读取本地配置，将公开模型的固定版本下载到 `models/`，不使用已保存的 Hugging Face 密钥。

在项目目录运行：

```powershell
.\scripts\install_local.ps1
```

虚拟环境和权重均被 Git 忽略。实测配置见 [本地模型说明](../docs/local_model.md)。

`summarize_chain.py` 读取探索运行记录，生成分组报告、Writer 样例及 Matplotlib
PNG/SVG 图，不调用模型。

`evaluate_chain.py` 在任务链暂停后，仅用本地模型执行完整的独立诊断任务，对比
冻结基座/当前 LoRA 及有/无当前记忆，不调用 Writer、不训练、不改变学习状态。

`validate_repairs.py` 将固定 Git 版本作为参照，检查同一记忆库下的检索消融、原始状态的
SFT 数据、实际样本覆盖和独立首次动作。输出写入新运行目录，不续跑原实验，不调用
Groq。仅训练模式支持显式指定固定数据、优化步数和 token 预算。
用 `--training-epochs 1` 可对旧运行的数据检查当前完整遍历策略；此选项与
`--training-steps` 互斥。

`install_native_linux.py` 把固定版本原生环境与校验和核验后的便携 Java 安装在项目内；`prepare_native_checks.py` 准备官方资源和明确的原生任务清单。`native_worker.py` 通过独立进程仅提供启动、动作和关闭接口；`check_native.py` 记录原生接口检查。

`check_mixed_sft.py` 检查本地语言模型的问答／决策混合目标、连续 LoRA 更新和适配器加载。问答属于诊断材料；此脚本不认证原生经验学习，也不退出记忆。见[方案落实](../docs/implementation_2026-10-24.zh-CN.md)。

`validate_repairs.py --legacy-projection` 才会显式复现已退出主路径的合成环境投射；默认保留来源状态。这是独立的历史动作诊断，不是当前混合训练协议。

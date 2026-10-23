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

`validate_repairs.py` 将固定 Git 版本作为参照，检查同一记忆库下的检索消融、重放校验的
SFT 数据、实际样本覆盖和独立首次动作。输出写入新运行目录，不续跑原实验，不调用
Groq。仅训练模式支持显式指定固定数据、优化步数和 token 预算。

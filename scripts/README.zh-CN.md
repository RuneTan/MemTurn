# 安装脚本

[English](README.md) | [简体中文](README.zh-CN.md)

`install_local.ps1` 创建 `.venv-local`，安装固定版本的 Windows CUDA 依赖并下载本地模型。脚本检查命令是否失败，不修改全局 Python。

`download_local_model.py` 读取本地配置，将公开模型的固定版本下载到 `models/`，不使用已保存的 Hugging Face 密钥。

在项目目录运行：

```powershell
.\scripts\install_local.ps1
```

虚拟环境和权重均被 Git 忽略。实测配置见 [本地模型说明](../docs/local_model.md)。

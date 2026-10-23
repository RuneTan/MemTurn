# 任务环境

[English](README.md) | [简体中文](README.zh-CN.md)

适配器实现 `reset(task)`、`step(action)` 和 `close()`；独立校验器检查来源是否支持记忆、训练目标是否有记录证据。

`toy.py` 提供两类随版本变化的有状态开发任务，公开动作词表和纠错反馈。任务模型读取公开观察，不读取校验器的答案表。

`scienceworld.py` 与 `alfworld.py` 隔离可选模拟器依赖。接口已用模拟环境检查，实际资源与基准校验器仍待接通；`webarena.py` 是暂缓实现的占位。

`verification.py` 拒绝未经认证的基准证据。任务最终成功不能自动认证全部中间步骤。环境资源与凭据不进入 Git。

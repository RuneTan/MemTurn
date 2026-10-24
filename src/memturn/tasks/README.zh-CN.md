# 任务环境

[English](README.md) | [简体中文](README.zh-CN.md)

适配器实现 `reset(task)`、`step(action)` 和 `close()`；独立校验器检查来源是否支持记忆、训练目标是否有记录证据。

`toy.py` 提供两类随版本变化的有状态开发任务，公开动作词表和纠错反馈。任务模型读取公开观察，不读取校验器的答案表。

可选的 `retrieval_scope(observation)` 提供公开的类型/版本兼容范围。历史投射逻辑只用于显式选择的旧诊断；当前校验器拒绝把后续动作移到更早输入的训练样本。

`scienceworld.py` 与 `alfworld.py` 隔离可选模拟器依赖。已增加隔离进程适配器以接通原生环境，领域证据校验仍待完成；`webarena.py` 是暂缓实现的占位。

`verification.py` 拒绝未经认证的基准证据。任务最终成功不能自动认证全部中间步骤。环境资源与凭据不进入 Git。

`process.py` 隔离模拟器与模型的运行依赖。`NativeReplayVerifier` 重放核查观察、反馈、原生证据、奖励与成功状态，缓存有上限；经验归纳还需独立领域判定。动作合法或最终成功都不直接认证任意摘要。见[落实状态](../../../docs/implementation_2026-10-24.zh-CN.md)。

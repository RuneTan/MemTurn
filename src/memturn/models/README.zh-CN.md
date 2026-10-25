# 模型后端

[English](README.md) | [简体中文](README.zh-CN.md)

`http.py` 提供远程聊天客户端、统一任务提示格式和 JSON 动作格式。各远程角色独立配置地址、模型及读取密钥的环境变量名。

任务提示展示已完成检索和剩余权限；不可用时不再提供检索操作说明，动作和查询字段
使用文字解释，避免模型照抄示例占位符。SFT 与推理使用同一格式。

`local.py` 加载本地 Transformers 权重，实际检查 CUDA 运算，并提供相同的 `complete()` 接口。生成与适配器切换串行执行；适配器使用独立版本名，固定的 Writer/Consolidator 继续调用基座。加载适配器不会将其合并进基座权重。

`agents.py` 提供提示式 Writer 与有限的只读 Consolidator 工具循环，输出必须通过独立来源校验。这尚不是 Auto-Dreamer 训练后的 Consolidator。

模型权重与具体部署逻辑均和记忆管理分离。

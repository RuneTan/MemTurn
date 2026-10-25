# 可选的外部 Writer

[English](writer_api.md) | [简体中文](writer_api.zh-CN.md)

以下提供商信息核对于 2026-10-23。更强的固定 Writer 可放在远端，任务模型和 LoRA 训练仍在本地。Writer 不需要可训练权重，其输出仍通过相同证据和样本校验；JSON 格式正确不等于有事实依据。

## 免费额度候选

| 提供商 | 候选 | 当时公开额度及适用范围 |
| --- | --- | --- |
| Groq | `openai/gpt-oss-120b` | 每分钟 30 次、每天 1000 次请求；每分钟 8000、每天 200000 tokens，适合小规模 Writer 接入检查 |
| Google AI Studio | `gemini-3.8-flash` | 列有免费输入／输出层；实际模型额度按项目在 AI Studio 查看 |
| OpenRouter | 当时可用的免费模型 | 免费计划列出每天 50 次请求，适合抽查，难以支撑反复实验 |

Groq 支持兼容聊天 API，GPT-OSS 120B 支持严格结构化输出。当前模板先使用 JSON object 模式，完整严格 schema 属于独立改进，也不能替代来源校验。Google 同样提供兼容聊天入口。免费可用性和额度可能变化，运行前须核实账户计划和具体模型。Google 价格页说明免费层内容可能用于改进产品，选择提交数据时应考虑这一点。

官方来源：

- [Groq 免费计划额度](https://console.groq.com/docs/rate-limits)
- [Groq 结构化输出](https://console.groq.com/docs/structured-outputs)
- [Groq 兼容接口](https://console.groq.com/docs/openai)
- [Gemini 价格](https://ai.google.dev/gemini-api/docs/pricing)
- [Gemini 当前项目额度](https://ai.google.dev/gemini-api/docs/rate-limits)
- [Gemini 兼容接口](https://ai.google.dev/gemini-api/docs/openai)
- [OpenRouter 价格](https://openrouter.ai/pricing)

## 配置

通过提供商控制台创建密钥，在启动进程中设置 `MEMTURN_WRITER_API_KEY`。程序只读指定环境变量，不打印其值；不会自动加载 `.env` 文件。两份 JSON 示例均不含密钥。

配置后先做两任务检查：

```text
.venv-local/Scripts/python.exe run_memturn.py local-smoke --tasks 2 --operators fixture --writer-profile configs/writer.groq.example.json
```

该命令组合本地任务模型、远程 Writer 和确定性 Consolidator。改用 `configs/writer.gemini.example.json` 可选择 Google。先检查服务可用性、实耗和 Writer 准入，再增加任务数和 `--train`。两个不同任务族的任务不保证能为合格 SFT 条目提供足够支持。

除非显式配置，HTTP 客户端不会发送 Qwen 专用的 `chat_template_kwargs`，避免将这些选项传给其他提供商。现有远程 Qwen 配置显式关闭 thinking。客户端发送应用 User-Agent，Writer 明确要求 JSON，以满足 Groq JSON object 模式要求；这两点已通过实际 API 检查。

## Groq 实测

运行 `local_20261023T151237196728Z` 组合了本地 Qwen3-1.7B、Groq GPT-OSS 120B Writer、确定性巩固及本地语言模型 SFT。四个任务完成，四个 Writer 输出全部入库，一次巩固和一次 LoRA 发布完成，无后台错误。Writer 有四次成功请求，使用 3561 个输入 tokens 和 1248 个输出 tokens；CUDA 分配峰值 9.09 GiB。adapter 在流末尾发布，未评估后续表现。Gemini 尚未实测。

应用通过环境变量读取凭据，不自动加载凭据文件。限流和提供商错误走后台失败记录，不会被悄悄转换成已接纳记忆。

# 配置

[English](README.md) | [简体中文](README.zh-CN.md)

`smoke.json` 用于标准库小环境；`local_qwen3_1_7b.json` 固定本地 GPU 模型、tokenizer 和 SFT 参数。ScienceWorld 与 ALFWorld 文件是尚待补齐的远程模板。

当前语言模型配置使用 `epochs: 1`、梯度累积为一：将选中条目的样本遍历一遍，每条
样本更新一次。token 和时限仍有效。显式 `steps` 保留给旧运行配置和历史诊断。

```text
python run_memturn.py check-profile --profile configs/scienceworld.example.json
```

各角色的服务地址与密钥从环境变量读取，密钥不写入 JSON。远程模板需要验证模型版本、LoRA 层、任务清单和证据校验器，初始到达策略会等待后台完成。

预算定义见 `src/memturn/config.py`。容量返还 token 比例固定为 0.5；`cold_fraction` 先确定冷记忆范围，再由 `region_token_cap` 限制实际输入。返还条数不单独设置。

本地 8,000 token 是暂定的接入预算，不是由显存推算或实验调优得到。每次检索最多
返回 512 token，来源/样本另有 32 MiB 上限。短 GPU 试跑尚未覆盖容量触顶，
详见 [预算依据](../docs/local_model.md)。

`writer.groq.example.json` 和 `writer.gemini.example.json` 为本地任务模型配置独立的
HTTP Writer。密钥只从指定的进程环境变量读取；Groq 已通过四任务接入检查，
Gemini 尚未验证，见 [接入说明](../docs/writer_api.md)。

`training_material` 默认为 `mixed`；`max_examples_per_job` 默认为 32，包含问答。`decision_only` 用于显式动作分类／历史诊断。`prepare_native_checks.py` 生成的原生进程清单与机器相关，只验证接口，不代表远程基准配置已完成。

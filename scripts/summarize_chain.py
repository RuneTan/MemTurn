"""Produce offline exploratory summaries and plots; never exposed to the agent."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def read_lines(path):
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return rows


def summarize(run, draw=True):
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    completed = report["next_index"]
    metrics = [r for r in read_lines(run / "metrics.jsonl") if r["completed"] <= completed]
    probes = list({(r["checkpoint"], r["condition"], r["sample"], r["regime"]): r
                   for r in read_lines(run / "probes.jsonl") if r["checkpoint"] <= completed}.values())
    groups = defaultdict(list)
    for row in metrics:
        groups[row["group"]].append(row)
    group_rows, previous_events = [], Counter()
    for group, rows in sorted(groups.items()):
        current = Counter(rows[-1]["events"])
        group_rows.append({"group": group + 1, "condition": rows[0]["condition"],
            "tasks": len(rows), "success_rate": statistics.mean(r["success"] for r in rows),
            "mean_reward": statistics.mean(r["reward"] for r in rows),
            "first_action_accuracy": statistics.mean(r["first_action_correct"] for r in rows),
            "mean_steps": statistics.mean(r["steps"] for r in rows),
            "writer_admitted": current["memory_written"] - previous_events["memory_written"],
            "publications": current["adapter_published"] - previous_events["adapter_published"]})
        previous_events = current
    probe_groups = defaultdict(list)
    for row in probes:
        probe_groups[(row["checkpoint"], row["regime"])].append(row["correct"])
    accuracy = [{"checkpoint": checkpoint, "regime": regime, "accuracy": statistics.mean(scores)}
                for (checkpoint, regime), scores in sorted(probe_groups.items())]
    condition_groups = defaultdict(list)
    for row in probes:
        if row["regime"] == "adapter_no_memory":
            condition_groups[(row["checkpoint"], row["condition"])].append(row["correct"])
    condition_accuracy = [{"checkpoint": checkpoint, "condition": condition, "accuracy": statistics.mean(scores)}
                          for (checkpoint, condition), scores in sorted(condition_groups.items())]
    summary = {"planned_tasks": report["target_tasks"], "completed_tasks": completed,
        "stop_reason": report["reason"], "success_rate": statistics.mean(r["success"] for r in metrics) if metrics else 0,
        "mean_reward": statistics.mean(r["reward"] for r in metrics) if metrics else 0,
        "first_action_accuracy": statistics.mean(r["first_action_correct"] for r in metrics) if metrics else 0,
        "writer_usage": report["writer_usage"], "writer_charged_tokens": report["writer_charged_tokens"],
        "events": report["events"], "groups": group_rows, "probe_accuracy": accuracy,
        "adapter_probe_by_condition": condition_accuracy,
        "peak_active_tokens": max((r["stats"]["active_tokens"] for r in metrics), default=0),
        "peak_source_training_bytes": max((r["stats"]["source_training_bytes"] for r in metrics), default=0),
        "peak_cuda_allocated_gib": max((r["peak_cuda_allocated_bytes"] for r in metrics), default=0) / 2**30,
        "peak_sampled_rss_gib": max((r["rss_after_maintenance_bytes"] for r in metrics), default=0) / 2**30,
        "max_adapter_weight_files": max((r["adapter_files"] for r in metrics), default=0),
        "elapsed_seconds": report["elapsed_seconds"],
        "limits": ["two synthetic task families and four known condition rules",
                   "fixture Consolidator; sequential maintenance; no independent evolving baseline",
                   "fixed probe objects are held out from writing/training; same-bank counterfactuals only",
                   "accepted-step training inputs can contain explicit corrective feedback",
                   "audit/report files are outside retrievable agent state"]}
    analysis = run / "analysis"
    analysis.mkdir(exist_ok=True)
    (analysis / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = read_lines(run / "audit.jsonl")
    sources = {r["task_id"]: r for r in audit if r["event"] == "task_completed"}
    candidates = [r for r in audit if r["event"] == "writer_candidate"]
    latest_task, null_cases = None, []
    expected = {"vault:v1": "inspect_lock", "api:v1": "refresh_token",
                "vault:v2": "check_permission", "api:v2": "renew_session"}
    for event in audit:
        if event["event"] == "task_completed":
            latest_task = event
        if event["event"] == "http_model_response" and latest_task:
            try:
                value = json.loads(event["text"])
            except (ValueError, TypeError):
                continue
            if value.get("memory") is None:
                null_cases.append({"task_id": latest_task["task_id"], "condition": latest_task["condition"],
                    "has_accepted_preparation": any(s["accepted"] and s["action"] == expected[latest_task["condition"]]
                                                    for s in latest_task["steps"])})
    failed = [r for r in audit if r["event"] == "task_completed" and not r["success"]]
    review = {"candidate_count": len(candidates), "null_count": len(null_cases),
              "null_with_accepted_preparation": sum(r["has_accepted_preparation"] for r in null_cases),
              "failed_tasks": len(failed), "failed_with_exhausted_search_actions": sum(
                  any(s["action"] == "invalid_memory_search_budget" for s in r["steps"]) for r in failed),
              "null_cases": null_cases, "association": "sequential foreground/Writer audit order"}
    summary["writer_review"] = {k: v for k, v in review.items() if k != "null_cases"}
    (analysis / "writer_review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    (analysis / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    by_condition = defaultdict(list)
    for candidate in candidates:
        by_condition[candidate["memory"]["applicability"]].append(candidate)
    samples = []
    for condition, rows in sorted(by_condition.items()):
        for row in (rows[:2] + rows[-2:] if len(rows) > 3 else rows):
            evidence = []
            for support in row["memory"]["supports"]:
                source = sources.get(support["task_id"], {})
                if 0 <= support["step"] < len(source.get("steps", [])):
                    evidence.append({"task_id": support["task_id"], "step": support["step"],
                                     **source["steps"][support["step"]]})
            samples.append({"condition": condition, "memory": row["memory"], "evidence": evidence})
    (analysis / "writer_samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 分组任务链探索性记录", "", f"计划 {report['target_tasks']} 个任务，实际完成 **{completed}** 个。停止状态：`{report['reason']}`。", "",
        f"总体完成率 {summary['success_rate']:.1%}，平均收益 {summary['mean_reward']:.3f}，首次动作正确率 {summary['first_action_accuracy']:.1%}。", "",
        f"Writer 成功调用 {report['writer_usage']['requests']} 次，按服务返回量扣除已报告缓存后累计 {report['writer_charged_tokens']:,} token。", "",
        f"维护后记忆正文采样最大值 {summary['peak_active_tokens']} token；GPU 分配峰值 {summary['peak_cuda_allocated_gib']:.2f} GiB。", "",
        "| 组 | 条件 | 任务数 | 完成率 | 平均收益 | 首次动作正确率 | 入库记忆 | 内化发布 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in group_rows:
        lines.append(f"| {row['group']} | {row['condition']} | {row['tasks']} | {row['success_rate']:.1%} | {row['mean_reward']:.3f} | {row['first_action_accuracy']:.1%} | {row['writer_admitted']} | {row['publications']} |")
    lines += ["", "首次动作探针在固定的未训练对象上运行，不提供纠错反馈、不写入经验，也不刷新记忆访问时间。四种设置分别固定基座或使用当前 LoRA，并关闭或读取当时的同一记忆库。这是局部诊断，不是四个独立演化的基线。", "",
              "这次只覆盖两类简单操作、四种带版本标签的规则。巩固仍由确定性的开发组件承担；任务间等待后台。数据不能证明跨领域持续学习、服务不停机或正式基准优势。", "",
              "训练样本来自已接受动作，其输入可能含环境直接给出的正确动作反馈；首次动作探针用于区分无反馈调用与跟随反馈的能力。", ""]
    if not report["events"].get("capacity_committed", 0):
        lines += ["本次尚未触发成功的容量维护，不能据此评估容量退出策略或检索多样性坍缩。", ""]
    final_file = analysis / "final_evaluation_summary.json"
    if final_file.exists():
        final = json.loads(final_file.read_text(encoding="utf-8"))
        summary["final_evaluation"] = final
        (analysis / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        lines += ["## 断点后完整任务诊断", "", "冻结基座与最终 LoRA 分别在同一组 8 个未训练对象上执行完整任务；不写入、不训练。这里只诊断当前模型和记忆状态。", "",
                  "| 设置 | 完成率 | 平均收益 |", "|---|---:|---:|"]
        for name, row in final.items():
            lines.append(f"| {name} | {row['success_rate']:.1%} | {row['mean_reward']:.3f} |")
        lines.append("")
    visible_file = analysis / "final_evaluation_tools_visible_summary.json"
    if visible_file.exists():
        visible = json.loads(visible_file.read_text(encoding="utf-8"))
        summary["final_evaluation_tools_visible"] = visible
        (analysis / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        lines += ["### 显式工具状态诊断", "", "仅补充剩余检索次数、上次检索查询和当前返回引用，不提示正确动作。使用相同最终权重与同一组任务，主任务链未采用此改动。", "",
                  "| 设置 | 完成率 | 平均收益 |", "|---|---:|---:|"]
        for name, row in visible.items():
            lines.append(f"| {name} | {row['success_rate']:.1%} | {row['mean_reward']:.3f} |")
        lines.append("")
    near_file = analysis / "final_evaluation_seed0_summary.json"
    if near_file.exists():
        near = json.loads(near_file.read_text(encoding="utf-8"))
        summary["final_evaluation_low_unseen_ids"] = near
        (analysis / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        lines += ["### 对象编号范围检查", "", "主探针使用百万量级编号；补充使用 0、1、10、11、20、21、30、31 这些未参与训练的编号，检查结论是否受编号范围影响。使用原有工具协议。", "",
                  "| 设置 | 完成率 | 首次决策正确率 |", "|---|---:|---:|"]
        for name, row in near.items():
            lines.append(f"| {name} | {row['success_rate']:.1%} | {row['first_decision_accuracy']:.1%} |")
        lines.append("")
    (analysis / "report.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")
    if draw and metrics:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
        colors = {"vault:v1": "#2563eb", "api:v1": "#16a34a", "vault:v2": "#9333ea", "api:v2": "#ea580c"}
        x = [r["group"] for r in group_rows]
        axes[0, 0].bar(x, [r["mean_reward"] for r in group_rows], color=[colors[r["condition"]] for r in group_rows])
        from matplotlib.patches import Patch
        axes[0, 0].legend(handles=[Patch(color=color, label=name) for name, color in colors.items()], fontsize=7)
        for row in group_rows:
            axes[0, 0].annotate(f"n={row['tasks']}", (row["group"], row["mean_reward"]),
                                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7)
        axes[0, 0].set(title="Mean reward by 20-task group", xlabel="Group", ylabel="Reward")
        axes[0, 1].plot(x, [r["success_rate"] for r in group_rows], "o-", label="Task success")
        axes[0, 1].plot(x, [r["first_action_accuracy"] for r in group_rows], "s-", label="First action")
        axes[0, 1].set(title="Task outcomes", xlabel="Group", ylim=(-.05, 1.05)); axes[0, 1].legend()
        axes[0, 0].set_xticks(x)
        axes[0, 1].set_xticks(x)
        for condition in colors:
            rows = [r for r in condition_accuracy if r["condition"] == condition]
            axes[1, 0].plot([r["checkpoint"] for r in rows], [r["accuracy"] for r in rows], "o-",
                            color=colors[condition], label=condition)
        macro = [r for r in accuracy if r["regime"] == "adapter_no_memory"]
        axes[1, 0].plot([r["checkpoint"] for r in macro], [r["accuracy"] for r in macro],
                        "k--", label="Macro average")
        axes[1, 0].set(title="Adapter-only held-out first-action accuracy", xlabel="Completed tasks", ylim=(-.05, 1.05))
        axes[1, 0].legend(fontsize=7)
        t = [r["completed"] for r in metrics]
        axes[1, 1].plot(t, [r["stats"]["active_tokens"] for r in metrics], label="Memory tokens after maintenance")
        axes[1, 1].set(title="Memory occupancy (configured cap: 8,000)", xlabel="Completed tasks", ylabel="Tokens")
        axes[2, 0].plot(t, [r["peak_cuda_allocated_bytes"] / 2**30 for r in metrics], label="Task CUDA peak")
        axes[2, 0].plot(t, [r["rss_after_maintenance_bytes"] / 2**30 for r in metrics], label="Sampled process RSS")
        axes[2, 0].set(title="Sampled resource use", xlabel="Completed tasks", ylabel="GiB"); axes[2, 0].legend()
        axes[2, 1].plot(t, [r["events"].get("adapter_published", 0) for r in metrics], label="Adapter publications")
        axes[2, 1].set(title="Cumulative adapter publications", xlabel="Completed tasks", ylabel="Publications")
        for axis in axes.flat: axis.grid(alpha=.2)
        figure.suptitle(f"MemTurn exploratory stream: {completed}/{report['target_tasks']} tasks\nSynthetic rules; fixture consolidation; sequential maintenance", fontsize=14)
        figure.savefig(analysis / "overview.png", dpi=160)
        figure.savefig(analysis / "overview.svg")
        plt.close(figure)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    summary = summarize(args.run.resolve(), not args.no_plot)
    print(json.dumps({"completed_tasks": summary["completed_tasks"], "stop_reason": summary["stop_reason"],
                      "analysis": str(args.run.resolve() / "analysis")}, ensure_ascii=False))

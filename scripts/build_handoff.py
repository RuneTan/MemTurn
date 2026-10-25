"""Allowlisted collaborator bundle, checksums and ZIP; never includes local secrets."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIRS = ('src','tests','scripts','configs','docs')
PROJECT_FILES = ('README.md','README.zh-CN.md','pyproject.toml','run_memturn.py','.gitignore','.env.example')
DESIGN_FILES = ('架构设计与搭建操作手册_2026-10-23.md','研究思路备忘录_2026-10-21.md',
                'MemTurn_架构一致性与工程审计_2026-10-24.md')
ALLOWED_SUFFIXES = {'.py','.md','.json','.txt','.toml','.sh','.ps1'}


def normalized_text(path):
    raw = Path(path).read_bytes()
    return raw.decode('utf-16') if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else raw.decode('utf-8-sig')


def write_evidence(source, destination):
    text = normalized_text(source)
    # Remove machine-specific workspace prefixes from shareable evidence only.
    for prefix in (str(ROOT), str(ROOT).replace('\\','\\\\'), ROOT.as_posix()):
        text = text.replace(prefix,'<LOCAL_MEMTURN_WORKSPACE>')
    destination.write_text(text,encoding='utf-8')


def build(output, name, summary_path):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundle = output/name
    bundle.mkdir(exist_ok=False)
    project = bundle/'MemTurn'
    project.mkdir()
    files = [ROOT/name for name in PROJECT_FILES if (ROOT/name).is_file()]
    for directory in PROJECT_DIRS:
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix in ALLOWED_SUFFIXES)
    for source in files:
        target = project/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    for name in DESIGN_FILES:
        shutil.copyfile(ROOT.parent/name,bundle/name)
    papers = ROOT.parent/'papers'
    if papers.exists():
        for source in papers.glob('*.md'):
            (bundle/'papers').mkdir(exist_ok=True)
            shutil.copyfile(source,bundle/'papers'/source.name)
    for name in ('架构设计与搭建操作手册_工程审计前_2026-10-24.md','研究思路备忘录_工程审计前_2026-10-24.md'):
        source = ROOT.parent/'过去版本存档'/name
        if source.exists():
            (bundle/'过去版本存档').mkdir(exist_ok=True)
            shutil.copyfile(source,bundle/'过去版本存档'/name)
    validation = bundle/'validation'
    validation.mkdir()
    evidence = {
        'windows-tests.log':ROOT/'tmp/handoff-windows-tests.log',
        'linux-tests.log':ROOT/'tmp/handoff-linux-tests.log',
        'local-inventory.json':ROOT/'tmp/handoff-local-inventory.json',
        'isolated-sft.json':ROOT/'runs/handoff_isolated_sft_metrics_20261024/report.json',
        'earlier-real-runtime.json':ROOT/'runs/audit_local_runtime_20261024/report.json',
        'earlier-mixed-sft.json':ROOT/'runs/audit_mixed_sft_20261024/report.json',
        'summary.json':Path(summary_path),
    }
    for name,source in evidence.items(): write_evidence(source,validation/name)
    summary = json.loads((validation/'summary.json').read_text(encoding='utf-8'))
    for key,name in (('windows','windows-tests.log'),('linux','linux-tests.log')):
        summary[key]['original_log_sha256'] = summary[key]['log_sha256']
        summary[key]['log_sha256'] = hashlib.sha256((validation/name).read_bytes()).hexdigest()
    summary['evidence_transform'] = 'logs converted to UTF-8 and local workspace prefixes replaced'
    (validation/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    readme = '''# MemTurn 合作研究交付包

工程验证日期：2026-10-24。供研究协作审阅和部署准备；正式模型、集群资源与实验方案尚未选定。没有执行远程部署或正式实验。

建议按以下顺序阅读：

1. [研究思路备忘录](研究思路备忘录_2026-10-21.md)：我们遇到什么问题，准备怎样解决，论文需要什么证据。
2. [架构手册](架构设计与搭建操作手册_2026-10-23.md)：系统的实际规则与首次运行步骤。
3. [当前审计](MemTurn_架构一致性与工程审计_2026-10-24.md)：问题如何修复，实现和验证各到哪一步。
4. [工程交接说明](MemTurn/docs/cluster_handoff.zh-CN.md)：安装、配置、调度与恢复细节；[验证摘要](validation/summary.json)及同目录报告保存实测证据。
5. [待确认清单](COLLABORATOR_CHECKLIST.zh-CN.md)：与合作者讨论硬件、模型、数据和正式实验时填写。

在包根目录先校验文件，再进入项目运行不需要模型的测试：

```bash
python MemTurn/scripts/verify_handoff.py .
cd MemTurn
python -X utf8 scripts/cluster_inventory.py --output cluster_inventory.json
python -X utf8 -m unittest discover -s tests -t . -v
python -X utf8 run_memturn.py demo --config configs/smoke.json --output runs/first_check
```

测试只用本机 HTTP 替身；缺少可选依赖时会跳过相应检查。模型、环境资产和 Python 环境须在集群单独安装。远程配置有意保留 TO_*，静态检查会以退出码 2 阻止未完成配置继续运行。

本包包含源码、研究设计文档、安装与调度模板、测试、故障恢复说明和经过筛选的验证报告；不包含密钥、Git 历史、模型权重、虚拟环境、完整任务轨迹或全部历史运行目录。MANIFEST.json 记录逐文件 SHA-256；它用于传输完整性检查，不是数字签名。请将整个目录或同名 ZIP 一并发送。

三份主文档已按当前实现精简同步；部分技术记录保留早期测试与修复历史，请留意其日期和历史说明。完整运行数据和权重不在包内，精选验证报告见 validation/。
'''
    (bundle/'README_FIRST.zh-CN.md').write_text(readme,encoding='utf-8')
    checklist = '''# 与合作者共同确认

以下均待讨论；没有预设服务器型号或正式模型。

| 项目 | 待填写内容 |
| --- | --- |
| 集群 | GPU 型号、数量、单卡显存、CPU/RAM、驱动/CUDA、调度器、最长作业时间、存储配额与锁/SQLite 支持 |
| 角色部署 | 任务模型、固定 Writer、固定 Consolidator；模型/tokenizer 不可变 revision，提示词及服务版本 |
| LoRA | 实际目标层、固定 rank/alpha、精度、训练与推理设备分配；是否需要当前实现尚未支持的多机训练 |
| 服务 | 到达间隔、并发/排队上限、任务时限、目标时延、拒绝和超时的计分 |
| 经验验证 | 首轮使用可严格检查的 observed_transition 子集，还是开发更广的领域谓词；保留原生反馈，不增加教师 |
| 任务流 | 环境实例与版本、共享结构/新变体/无关任务/旧情境返回安排、开发/测试划分、流长度、随机种子 |
| 预算 | 全部经验状态、主库及辅助文件、模型版本、RAM/GPU、各角色 token、训练 token/更新、总运行时间 |
| 对照 | 从已实现的十组中选择正式组合；明确同上限比较与同实际训练量比较的区别 |
| Auto-Dreamer | 核实可获取的原代码/训练后权重及许可；否则继续明确标注提示原型 |
| 验收 | 固定模型可推理/训练、领域反例检查、至少三批延续、容量压力、超时/重启/过载、真实服务缓存释放及长程统计 |

建议先返还 cluster_inventory.json 和集群限制，完成小规模接入验收，再在开发集校准数值并冻结正式方案。已有软件测试及两轮 SFT 记录不能替代上述集群和科学验收。
'''
    (bundle/'COLLABORATOR_CHECKLIST.zh-CN.md').write_text(checklist,encoding='utf-8')
    suspicious = []
    patterns = (re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
                re.compile(rb'\b(?:sk-proj-|gsk_|hf_)[A-Za-z0-9_-]{40,}'))
    for path in bundle.rglob('*'):
        if path.is_file() and any(pattern.search(path.read_bytes()) for pattern in patterns):
            suspicious.append(path.relative_to(bundle).as_posix())
    if suspicious: raise ValueError('bundle secret-pattern scan requires review: '+', '.join(suspicious))
    entries = []
    for path in sorted(bundle.rglob('*')):
        if path.is_file():
            data = path.read_bytes()
            entries.append({'path':path.relative_to(bundle).as_posix(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    manifest = {'created_utc':datetime.now(timezone.utc).isoformat(),'base_commit':head,
                'source_kind':'working_tree_snapshot',
                'secret_scan':'allowlist_and_pattern_scan_passed','files':entries}
    (bundle/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    archive = output/(bundle.name+'.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as zipped:
        for path in sorted(bundle.rglob('*')):
            if path.is_file(): zipped.write(path,path.relative_to(output).as_posix())
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output/(archive.name+'.sha256')).write_text(checksum+'  '+archive.name+'\n',encoding='ascii')
    return {'directory':str(bundle),'zip':str(archive),'sha256':checksum,'files':len(entries),'bytes':archive.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--name',default='MemTurn_handoff_2026-10-24')
    parser.add_argument('--summary',required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output,args.name,args.summary),ensure_ascii=False,indent=2))


if __name__ == '__main__': main()

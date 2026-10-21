# -*- coding: utf-8 -*-
"""Qwen3-VL 全量管线健康检查（远程统计，不落地任何数据文件）。

只回答「有没有系统性崩」，不解读为最终效果——部分数据的成功率不可外推。

检查项：
- 进度 / 日志新鲜度（停滞检测）
- **进程与 GPU 占用**：多于 1 个 python 推理进程即告警——并发写同一 JSON 会
  互相覆盖，且抢占 GPU 使吞吐腰斩（实测 0.26 → 0.41 q/s）
- direct：条数、命中率、非法框（逆序/越界）、raw 有框却未解析、每图框数分布
- som：条数、reason 分布、n_cands 分布、有框数
- 日志中 [ERR] 行数、启动次数

用法：
    设好 SSH_HOST/SSH_PORT/SSH_USER/SSH_PW 后直接运行
"""
import os

import paramiko

CMD = r'''
echo ==PROGRESS==
tail -n 3 /srv/output/qwen_full.log
echo ==PROCESSES==
ps -eo pid,lstart,etimes,cmd | grep -E "[p]ython -u" || echo "(无)"
echo ==GPU_APPS==
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo ==HEALTH==
stat -c "日志静默 %Y" /srv/output/qwen_full.log
date +%s
echo -n "[ERR] 行数: "; grep -c "\[ERR\]" /srv/output/qwen_full.log || true
echo -n "启动次数: "; grep -c "图 2000" /srv/output/qwen_full.log || true
echo ==STATS==
/srv/venv-locate/bin/python - <<'PY'
import json
import re
from collections import Counter

RE = re.compile(r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')
d = json.load(open('/srv/output/qwen3vl_all.json', encoding='utf-8'))
s = json.load(open('/srv/output/qwen3vl_som_all.json', encoding='utf-8'))

nd = len(d)
hit = sum(1 for v in d.values() if v.get('bbox'))
print(f'direct {nd}  有框 {hit} ({hit / max(nd, 1):.1%})')

bad = [k for k, v in d.items() if v.get('bbox') and not (
    0 <= v['bbox'][0] < v['bbox'][2] <= 1 and 0 <= v['bbox'][1] < v['bbox'][3] <= 1)]
print(f'非法框(逆序/越界) {len(bad)}  {bad[:6]}')

swapped = [k for k, v in d.items() if v.get('bbox') and (
    v['bbox'][0] > v['bbox'][2] or v['bbox'][1] > v['bbox'][3])]
oob = [k for k, v in d.items() if v.get('bbox') and (
    not (0 <= v['bbox'][0] <= 1 and 0 <= v['bbox'][1] <= 1
         and 0 <= v['bbox'][2] <= 1 and 0 <= v['bbox'][3] <= 1))]
print(f'  其中逆序 {len(swapped)} / 越界 {len(oob)}')

unparsed = [k for k, v in d.items()
            if not v.get('bbox') and RE.search(v.get('raw') or '')]
print(f'raw 含框却未解析 {len(unparsed)}  {unparsed[:6]}')
print(f'每图解析框数 n 分布 {dict(sorted(Counter(v.get("n") for v in d.values()).items(), key=lambda x: (x[0] is None, x[0])))}')

ns = len(s)
sh = sum(1 for v in s.values() if v.get('bbox'))
print(f'som {ns}  有框 {sh} ({sh / max(ns, 1):.1%})')
print(f'som reason {dict(Counter(v.get("reason", "ok") for v in s.values()))}')
nc = Counter(v.get('n_cands') for v in s.values())
print(f'som n_cands {dict(sorted(nc.items(), key=lambda x: (x[0] is None, x[0])))}')
weak = sum(1 for v in s.values() if (v.get('n_cands') or 0) <= 1)
print(f'  枚举弱(n_cands<=1) {weak}  枚举出多候选 {ns - weak}')
PY
echo ==GPU==
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
'''


def main() -> int:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(os.environ["SSH_HOST"], port=int(os.environ["SSH_PORT"]),
                username=os.environ["SSH_USER"],
                password=os.environ["SSH_PW"], timeout=25,
                banner_timeout=30, auth_timeout=30)
    _, o, _ = cli.exec_command(CMD, timeout=180)
    out = o.read().decode("utf-8", "replace")
    cli.close()
    print(out)

    # 按 ==SECTION== 分段解析：不能整段文本里找「数字开头」，因为 `date +%s`
    # 输出的裸 epoch 也会被误计为进程行。
    sec: dict = {}
    cur = None
    for ln in out.splitlines():
        if ln.startswith("==") and ln.rstrip().endswith("=="):
            cur = ln.strip().strip("=").strip()
            sec[cur] = []
        elif cur:
            sec[cur].append(ln)
    procs = [ln for ln in sec.get("PROCESSES", []) if "python -u" in ln]
    gpu = [ln for ln in sec.get("GPU_APPS", [])
           if ln.strip() and not ln.lower().startswith("pid")]
    # 并发写同一 JSON 是隐性灾难：两边都是「启动时载入、整体覆盖写」，不合并。
    print(f"[判读] 推理进程 {len(procs)} 个 / GPU 计算进程 {len(gpu)} 个", end="")
    if len(procs) > 1 or len(gpu) > 1:
        print("  *** 告警：存在并发写入者，吞吐与产物都在被污染 ***")
    else:
        print("  —— 正常")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

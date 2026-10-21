#!/bin/bash
# VLX-Seek：取真实调用流程 + 冻结冒烟清单 + 重试 flash_attn + 查权重进度
set -u
R=/root/rivermind-data
VLX=$R/vlx-seek
RES=$VLX/resources

echo "=== A. inference.py 里检测器如何被调用（★驱动的依据） ==="
grep -nE "WeDetect|generate_proposal|detector|proposal|bbox_list|worker\.|run_task|ground|task ==" \
  "$VLX/inference.py" 2>/dev/null | head -40 | sed 's/^/  /'

echo; echo "=== B. generate_proposal.py 的对外入口 ==="
grep -nE "^def [a-z]|^class [A-Z]|if __name__|return .*boxes|def .*proposal" \
  "$VLX/detect_tools/WeDetect/generate_proposal.py" 2>/dev/null | tail -25 | sed 's/^/  /'

echo; echo "=== C. 任务模板（grounding_single 的提示词长什么样） ==="
grep -nE "grounding_single|TASK|template" "$VLX/vlx_seek/task_templates.py" 2>/dev/null \
  | head -20 | sed 's/^/  /'

echo; echo "=== D. 重试 flash_attn（--no-build-isolation，后台） ==="
if [ -x "$R/venv_vlx/bin/python" ]; then
  nohup "$R/venv_vlx/bin/pip" install --no-build-isolation "flash_attn==2.8.3" \
    > "$R/vlx_flash.log" 2>&1 &
  echo "  已后台启动 PID=$!"
fi

echo; echo "=== E. 冻结冒烟清单（20 条：序数 10 / 非序数 10，按 qid 排序取，确定性） ==="
"$R/venv_vlx/bin/python" - <<'PY' 2>&1 | sed 's/^/  /'
import json, os
R = "/root/rivermind-data"
q = json.load(open(f"{R}/dataset/queries/queries.json", encoding="utf-8"))
cores = set(json.load(open(f"{R}/_ordinal_cores_r2.json", encoding="utf-8")))
have = set(os.listdir(f"{R}/dataset/Images/visible"))
ordi, non = [], []
for k in sorted(q):
    if os.path.basename(q[k]["visible"]) not in have:
        continue
    (ordi if k in cores else non).append(k)
sel = non[:10] + ordi[:10]
sel = sorted(sel)
out = f"{R}/_vlx_smoke20.json"
json.dump(sel, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"可读图条目：非序数 {len(non)} / 序数 {len(ordi)}")
print(f"清单已冻结 {out}：{len(sel)} 条（非序数 10 + 序数 10）")
for k in sel[:4]:
    print(f"   {k}  {q[k]['query'][:56]}")
PY

echo; echo "=== F. 权重进度 ==="
pgrep -f "_vlx_dl_model" >/dev/null && echo "  ● 仍在下" || echo "  ○ 已结束"
du -sh "$RES/VLX-Seek-1.5-10B" 2>/dev/null | sed 's/^/  /'
tail -2 "$R/vlx_dl_model.log" 2>/dev/null | sed 's/^/  /'
echo "=== 结束 ==="

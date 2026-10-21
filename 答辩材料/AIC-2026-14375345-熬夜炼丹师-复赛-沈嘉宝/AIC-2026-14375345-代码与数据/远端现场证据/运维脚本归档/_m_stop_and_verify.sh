#!/bin/bash
# 停掉跑错配置的全量 + 核查远端脚本版本 + 清掉错产物
set -u
R=/root/rivermind-data

echo "=== ① 停掉跑错配置的全量 ==="
pkill -f "_qwen_full_pipeline.py" 2>/dev/null
sleep 3
if pgrep -af "_qwen_full_pipeline.py" >/dev/null 2>&1; then
  pgrep -af "_qwen_full_pipeline.py" | cut -c1-140 | sed 's/^/  ★残留 /'
else
  echo "  ✓ 已停"
fi

echo
echo "=== ② 核查远端 _launch_q38.sh（是否含 QWEN_EXTRA 透传）==="
if grep -q "QWEN_EXTRA" "$R/_launch_q38.sh" 2>/dev/null; then
  echo "  ✓ 含 QWEN_EXTRA"
else
  echo "  ★远端仍是旧版（无 QWEN_EXTRA）⇒ 必须重传"
fi
grep -n "QWEN_EXTRA\|QWEN_ONLY_NONORD=\|PY=\${QWEN_PY" "$R/_launch_q38.sh" 2>/dev/null | sed 's/^/    /'

echo
echo "=== ③ 核查其余文件的关键特征 ==="
echo "  _qwen_full_pipeline.py : merge3=$(grep -c 'def merge3' "$R/_qwen_full_pipeline.py" 2>/dev/null || echo 0) MERGED_PROMPT=$(grep -c 'MERGED_PROMPT' "$R/_qwen_full_pipeline.py" 2>/dev/null || echo 0) remap_merged=$(grep -c 'def remap_merged' "$R/_qwen_full_pipeline.py" 2>/dev/null || echo 0)"
echo "  _q38_gate.py           : 新打印行=$(grep -c 'multi_box=' "$R/_q38_gate.py" 2>/dev/null || echo 0)  merge3 用法=$(grep -c 'P.merge3' "$R/_q38_gate.py" 2>/dev/null || echo 0)"
echo "  _launch_M.sh           : MERGE_SPACE=$(grep -c 'QWEN_MERGE_SPACE' "$R/_launch_M.sh" 2>/dev/null || echo 0)"

echo
echo "=== ④ 清掉错配置的产物（只覆盖了 1150 图的一个子集）==="
ls -l "$R/output/m_direct.json" 2>/dev/null | sed 's/^/  删除 /'
rm -f "$R/output/m_direct.json" "$R/output/m_som.json" "$R/m_run.log"
echo "  ✓ 已清理"

echo
echo "=== ⑤ 顺带确认服务仍在 ==="
curl -s --max-time 6 http://127.0.0.1:8000/v1/models 2>/dev/null | tr ',' '\n' | grep '"id"' | sed 's/^/  /'
echo "=== DONE ==="

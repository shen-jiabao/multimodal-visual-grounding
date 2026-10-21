#!/bin/bash
# M 件改动后的语法检查 + M 配置下的模块级自检（导入即触发断言）
set -u
. /root/rivermind-data/_q38_env.sh
R=/root/rivermind-data
PY=$R/venv_q38/bin/python

echo "=== ① 语法检查 ==="
for f in _qwen_full_pipeline.py _q38_gate.py _m_thinking_probe.py \
         _prereg_M_selftest.py; do
  if $PY -m py_compile "$R/$f" 2>/tmp/_syn.err; then
    echo "  ✓ $f"
  else
    echo "  ★ $f 语法错误："; sed 's/^/      /' /tmp/_syn.err
  fi
done

echo
echo "=== ② shell 语法检查 ==="
for f in _launch_q38.sh _launch_M.sh _q38_gate.sh _get_serve_q38.sh; do
  if bash -n "$R/$f" 2>/tmp/_syn2.err; then echo "  ✓ $f"; else
    echo "  ★ $f："; sed 's/^/      /' /tmp/_syn2.err; fi
done

echo
echo "=== ③ M 配置下导入（模块级断言会在此触发）==="
cd "$R"
QWEN_MODALITIES=3m QWEN_RULES5=1 QWEN_MULTIBOX=1 QWEN_MAXNEW=1024 \
QWEN_THINKING=0 QWEN_ENGINE=api32b \
  $PY -c "
import _qwen_full_pipeline as P
print('  导入成功')
print('  MODALITIES', P.MODALITIES, '| MAXBOX', P.MAXBOX, '| MAXNEW', P.MAXNEW,
      '| THINKING', P.THINKING, '| RULES5', P.RULES5)
print('  RULES_BLOCK 长度', len(P.RULES_BLOCK))
print('  MERGED_PROMPT 长度', len(P.MERGED_PROMPT),
      '| MULTIBOX_PROMPT 长度', len(P.MULTIBOX_PROMPT))
print('  MERGE_SPACE', P.MERGE_SPACE, '| scene_where 目标说法:',
      P.scene_where()[1][:46])
print('  CONC', P.CONC, '| PHASE', P.PHASE)
"

echo
echo "=== ④ 非法组合必须被断言拦住（应看到 AssertionError）==="
QWEN_ENGINE=api32b QWEN_MULTIBOX=1 QWEN_MODALITIES=1 $PY -c "
import _qwen_full_pipeline as P" 2>&1 | tail -3 | sed 's/^/  /'
echo "=== DONE ==="

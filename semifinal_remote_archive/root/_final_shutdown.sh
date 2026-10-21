#!/bin/bash
# 关服前收尾：确认无在跑任务 + 生成关服状态清单（含关键文件哈希）
set -u
R=/root/rivermind-data
OUT=$R/_shutdown_manifest.txt
{
  echo "# 关服状态清单   $(date)"
  echo
  echo "## 1) 在跑进程（应只剩 vLLM 服务本体）"
  pgrep -af '_qwen_full_pipeline.py' | grep -v 'bash -c' || echo "  ✓ 无管线在跑"
  echo "  vLLM: $(pgrep -f 'vllm serve' | head -1 | sed 's/^/PID=/')"
  echo
  echo "## 2) GPU"
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader | sed 's/^/  /'
  echo
  echo "## 3) 磁盘"
  df -h "$R" | tail -1 | sed 's/^/  /'
  echo
  echo "## 4) 关键文件哈希"
  for f in _qwen_full_pipeline.py _d_scope.json _d_sample300.json \
           _ordinal_cores_v3.json output/qwen32b_direct.json \
           output/qwen32b_direct_ord.json output/qwen32b_som_scoped.json \
           output/qwen32b_d3_full.json output/qwen32b_d3_sample.json ; do
    if [ -f "$R/$f" ]; then
      printf "  %s  %s\n" "$(sha256sum "$R/$f" | cut -c1-64)" "$f"
    else
      printf "  %-64s %s\n" "(不存在)" "$f"
    fi
  done
  echo
  echo "## 5) 远端脚本目录"
  ls -1 "$R"/*.sh "$R"/*.py 2>/dev/null | sed 's|.*/|  |'
  echo
  echo "## 6) 日志"
  ls -1 "$R"/*.log "$R"/output/*.log 2>/dev/null | sed 's|.*/|  |' || echo "  (无)"
} > "$OUT" 2>&1
cat "$OUT"
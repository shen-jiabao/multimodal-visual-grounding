#!/bin/bash
# VLX-Seek 准备状态总览（纯 bash，经 _rc.py --script 走 stdin，避开本地引号）
set -u
R=/root/rivermind-data
VLX=$R/vlx-seek
RES=$VLX/resources
D=$RES/VLX-Seek-1.5-10B
F=$D/model.safetensors
EXP=20033961788          # 官方 model.safetensors 字节数（HEAD 实测）
PY=$R/venv_vlx/bin/python

echo "=== 1. 权重下载（目标 ${EXP} 字节 ≈ $((EXP/1048576)) MB） ==="
s1=$(stat -c%s "$F" 2>/dev/null || echo 0)
sleep 12
s2=$(stat -c%s "$F" 2>/dev/null || echo 0)
rate=$(awk -v a="$s1" -v b="$s2" 'BEGIN{printf "%.2f", (b-a)/12/1048576}')
echo "  当前  $((s2/1048576)) MB / $((EXP/1048576)) MB"
echo "  覆盖  $(awk -v a="$s2" -v b="$EXP" 'BEGIN{printf "%.2f", 100*a/b}') %"
echo "  速率  ${rate} MB/s（12 秒采样）"
[ "$s2" -ge "$EXP" ] && echo "  ● 已完成（大小一致）" || echo "  ○ 未完成"
pgrep -f "_vlx_dl2" >/dev/null && echo "  进程  ● 续传循环在跑" || echo "  进程  ○ 循环已退出"
tail -4 "$R/vlx_dl2.log" 2>/dev/null | sed 's/^/  log| /'

echo; echo "=== 2. 权重目录内容 ==="
ls -1 "$D" 2>/dev/null | sed 's/^/  /' || echo "  （目录不存在）"

echo; echo "=== 3. WeDetect 检测器 ==="
if [ -f "$RES/wedetect_base_uni.pth" ]; then
  echo "  ● $(du -h "$RES/wedetect_base_uni.pth" | cut -f1)  $(stat -c%s "$RES/wedetect_base_uni.pth") 字节"
else
  echo "  ○ 缺失"
fi

echo; echo "=== 4. venv 与依赖 ==="
if [ -x "$PY" ]; then
  echo "  ● venv 存在：$PY"
  "$PY" - <<'PYEOF' 2>&1 | sed 's/^/  /'
import importlib
for m in ("torch", "torchvision", "transformers", "timm", "einops",
          "accelerate", "flash_attn"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:14s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:14s} 【缺】{type(e).__name__}")
try:
    import torch
    print(f"  cuda_available {torch.cuda.is_available()}")
except Exception as e:
    print(f"  cuda 查询失败 {e}")
PYEOF
else
  echo "  ○ venv 缺失：$PY"
fi

echo; echo "=== 5. 源码完整性与 import ==="
n=$(find "$VLX" -name '*.py' 2>/dev/null | wc -l)
e=$(find "$VLX" -name '__init__.py' -size 0 2>/dev/null | wc -l)
echo "  .py 文件数 $n ；空 __init__.py 数 $e"
[ -x "$PY" ] && (cd "$VLX" && "$PY" - <<'PYEOF' 2>&1 | tail -3 | sed 's/^/  /'
import vlx_seek_worker as w
print("  import vlx_seek_worker OK ->", getattr(w, "VLXSeekWorker", "?"))
PYEOF
) || echo "  ○ 跳过 import 自检"

echo; echo "=== 6. 驱动与冒烟清单 ==="
for f in _vlx_direct.py _vlx_smoke20.json; do
  p=$R/$f
  if [ -f "$p" ]; then
    echo "  ● $f  $(stat -c%s "$p") 字节"
  else
    echo "  ○ 缺 $f"
  fi
done

echo; echo "=== 7. GPU 与磁盘 ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
           --format=csv,noheader 2>/dev/null | sed 's/^/  /' \
  || echo "  ○ 无 GPU（nvidia-smi 不可用）"
ls /dev/nvidia* 2>/dev/null | sed 's/^/  dev /' || echo "  dev: 无 /dev/nvidia*"
df -h "$R" | tail -1 | sed 's/^/  /'
echo "=== 结束 ==="

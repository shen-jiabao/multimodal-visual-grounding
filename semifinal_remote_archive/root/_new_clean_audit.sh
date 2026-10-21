#!/bin/bash
# 干净度深查：本机是否只服务 Qwen3.8-27B 一件事
set -u
. /root/rivermind-data/_q38_env.sh
R=/root/rivermind-data

echo "=== ① 本机根目录一览 ==="
ls -la /root/ 2>/dev/null | sed 's/^/  /'
echo "  --- /root/autodl-tmp（数据盘实体）---"
ls -la /root/autodl-tmp/ 2>/dev/null | sed 's/^/  /'

echo
echo "=== ② 数据盘上除目标模型外，还有无其他模型权重 ==="
find /root/autodl-tmp -xdev \( -name "*.safetensors" -o -name "*.gguf" -o -name "*.pth" -o -name "*.bin" \) 2>/dev/null \
  | grep -v "/models/Qwen3.8-27B/" | head -20 | sed 's/^/  /'
echo "    ↑ 空 = 只有目标模型"
echo "  --- 其他 HF 缓存 ---"
du -sh /root/autodl-tmp/rivermind-data/hf 2>/dev/null | sed 's/^/  /'

echo
echo "=== ③ 临时目录明细（判断哪些是死数据）==="
for p in "$R/tmp" "$R/pipcache" "$R/uvcache" "$R/cache"; do
  echo "  --- $p ($(du -sh "$p" 2>/dev/null | cut -f1)) ---"
  ls -1 "$p" 2>/dev/null | head -8 | sed 's/^/    /'
done

echo
echo "=== ④ 分片残片是否清干净 ==="
echo "  .parts 内容: $(ls -1 "$R/models/Qwen3.8-27B/.parts" 2>/dev/null | wc -l) 个"
echo "  模型目录文件数: $(ls -1 "$R/models/Qwen3.8-27B" | wc -l)"

echo
echo "=== ⑤ 环境文件（新机统一入口）内容 ==="
cat "$R/_q38_env.sh" 2>/dev/null | sed 's/^/  /'

echo
echo "=== ⑥ 无卡模式判定 ==="
if ls /dev/nvidia[0-9]* >/dev/null 2>&1; then echo "  有 /dev/nvidia* ⇒ 已切有卡"; else echo "  无 /dev/nvidia* ⇒ 仍为无卡模式"; fi
df -h / /root/autodl-tmp 2>/dev/null | sed 's/^/  /'
echo "=== DONE ==="

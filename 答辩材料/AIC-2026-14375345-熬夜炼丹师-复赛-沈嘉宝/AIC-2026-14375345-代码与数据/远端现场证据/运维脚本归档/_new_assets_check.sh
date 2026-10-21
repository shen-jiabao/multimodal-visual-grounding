#!/bin/bash
# 新机就绪盘点：数据 / 查询表 / 推理脚本 / 权重 是否齐备
set -u
R=/root/rivermind-data
D=$R/dataset

echo "=== ① 数据集 ==="
for m in visible infrared depth; do
  n=$(ls -1 "$D/Images/$m" 2>/dev/null | wc -l)
  echo "  Images/$m: $n 文件"
done
echo "  --- 其余数据集内容 ---"
find "$D" -maxdepth 2 -not -path "*/Images/*" 2>/dev/null | head -20 | sed 's/^/    /'
for f in "$D/queries/queries.json" "$R/dataset/queries/queries.json"; do
  [ -f "$f" ] && echo "  ✓ $f  ($(stat -c%s "$f") B)" || echo "  ★缺 $f"
done

echo
echo "=== ② 推理所需脚本/中间件 ==="
for f in _qwen_full_pipeline.py _q38_gate.py _probe_qids.json \
         _ordinal_cores_r2.json _imglist.txt _q38_env.sh; do
  if [ -f "$R/$f" ]; then echo "  ✓ $f  ($(stat -c%s "$R/$f") B)"; else echo "  ★缺 $f"; fi
done

echo
echo "=== ③ 权重 ==="
W=$R/models/Qwen3.8-27B
echo "  完整分片: $(ls -1 "$W"/model-*-of-*.safetensors 2>/dev/null | wc -l) / 18"
echo "  目录体量: $(( $(du -sb "$W" 2>/dev/null | awk '{print $1}') / 1048576 )) MB"
ls -1 "$W"/*.json "$W"/*.txt 2>/dev/null | head -8 | sed 's/^/    /'

echo
echo "=== ④ venv 自证 ==="
$R/venv_q38/bin/python -c "
import vllm, torch, openai, PIL
print('  vllm', vllm.__version__, '| torch', torch.__version__)
print('  openai', openai.__version__, '| pillow', PIL.__version__)
" 2>&1 | sed 's/^/  /'
echo "=== DONE ==="

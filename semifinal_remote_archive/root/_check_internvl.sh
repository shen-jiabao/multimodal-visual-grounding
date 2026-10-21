#!/bin/bash
# InternVL 路线可行性核查（不需要 GPU）
#   1) vLLM 0.11.0 里 InternVL 的实现是否覆盖 InternVL3.5
#   2) 量化版（AWQ/INT4）是否发布 —— 决定单卡能否跑 38B
set -u
V=/root/rivermind-data/venv/bin/python

echo "=== 1) vLLM 里的 InternVL 实现 ==="
SP=$("$V" -c "import vllm,os;print(os.path.dirname(vllm.__file__))" 2>/dev/null)
echo "  vllm 包路径: $SP"
echo "  --- models 目录下 intern* 文件 ---"
ls -1 "$SP/model_executor/models/" 2>/dev/null | grep -i intern | sed 's/^/    /'
echo "  --- internvl.py 里与"版本/架构"有关的行 ---"
for f in internvl.py internvl2.py; do
  [ -f "$SP/model_executor/models/$f" ] || continue
  echo "    [文件] $f"
  grep -nEi "InternVL3|internvl3|visual.?resolution|lang_model_name|architectures|model_type" \
    "$SP/model_executor/models/$f" 2>/dev/null | head -12 | sed 's/^/      /'
done
echo "  --- registry 中全部 Intern 架构 ---"
"$V" - <<'PYEOF' 2>/dev/null
from vllm.model_executor.models.registry import ModelRegistry
reg = ModelRegistry.get_supported_archs()
for k in sorted(reg):
    if "Intern" in k:
        print(f"      {k}  ->  {reg[k]}")
PYEOF

echo
echo "=== 2) 量化版可用性（ModelScope / HF）==="
for repo in InternVL3_5-38B InternVL3_5-38B-AWQ InternVL3_5-38B-Int4 \
            InternVL3_5-38B-GPTQ InternVL3-38B-AWQ InternVL3_5-8B-AWQ ; do
  hf=$(timeout 18 curl -s -o /dev/null -w '%{http_code}' \
        "https://huggingface.co/api/models/OpenGVLab/$repo" 2>/dev/null || echo 000)
  ms=$(timeout 18 curl -s -o /dev/null -w '%{http_code}' \
        "https://modelscope.cn/api/v1/models/OpenGVLab/$repo" 2>/dev/null || echo 000)
  printf "  %-24s HF=%s  ModelScope=%s\n" "$repo" "$hf" "$ms"
done

echo
echo "=== 3) HF 上 OpenGVLab 的 InternVL3_5 系列全列 ==="
timeout 25 curl -s "https://huggingface.co/api/models?author=OpenGVLab&search=InternVL3_5&limit=100" \
  2>/dev/null | "$V" -c "
import json,sys
try:
    d = json.load(sys.stdin)
    for m in d: print('    ', m.get('modelId'))
except Exception as ex:
    print('     解析失败:', type(ex).__name__)
" 2>/dev/null

echo
echo "=== 4) 磁盘再确认 ==="
df -h /data /root/rivermind-data 2>/dev/null | sed 's/^/  /'
echo "=== 核查结束 ==="
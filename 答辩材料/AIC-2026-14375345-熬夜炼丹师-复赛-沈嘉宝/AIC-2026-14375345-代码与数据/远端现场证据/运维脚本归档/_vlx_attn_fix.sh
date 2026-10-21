#!/bin/bash
# 打 attention 兼容补丁（env 可覆盖，默认行为不变）+ 探 flash_attn 安装可行性
set -u
VLX=/root/rivermind-data/vlx-seek
PY=/root/rivermind-data/venv_vlx/bin/python

echo "=== 1. 打补丁（原文件哈希先留痕，可回退） ==="
"$PY" - <<'PYEOF'
import hashlib
from pathlib import Path

BASE = Path("/root/rivermind-data/vlx-seek/vlx_seek/models/vlx_seek_1_5")
HEADER = (
    "import os as _os\n"
    '# 兼容补丁：官方硬编码 flash_attention_2；缺 flash_attn 时可设\n'
    '# VLX_ATTN=sdpa 回退。默认值即官方行为，不设环境变量则行为不变。\n'
    '_VLX_ATTN = _os.environ.get("VLX_ATTN", "flash_attention_2")\n'
)

for rel in ("builder.py", "multimodal_encoder/qwen3_5_vl_encoder.py"):
    p = BASE / rel
    raw = p.read_bytes()
    print(f"  {rel}")
    print(f"    原 sha256 {hashlib.sha256(raw).hexdigest()}")
    s = p.read_text(encoding="utf-8")
    if "_VLX_ATTN" in s:
        print("    已打过补丁，跳过")
        continue
    n = s.count('"flash_attention_2"')
    s2 = s.replace('"flash_attention_2"', "_VLX_ATTN")
    # 插到 __future__ 之后（若有），否则插到最前
    lines = s2.split("\n")
    at = 0
    for i, ln in enumerate(lines[:8]):
        if ln.startswith("from __future__"):
            at = i + 1
    lines[at:at] = ["", *HEADER.rstrip("\n").split("\n")]
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"    替换 {n} 处；新 sha256 "
          f"{hashlib.sha256(p.read_bytes()).hexdigest()}")
    # 回显改动行，供人工核对
    for i, ln in enumerate(p.read_text(encoding="utf-8").split("\n"), 1):
        if "_VLX_ATTN" in ln and "os.environ" not in ln:
            print(f"      L{i}: {ln.strip()}")
PYEOF

echo; echo "=== 2. 补丁后自检（不设环境变量 ⇒ 仍要 flash_attn；设 sdpa ⇒ 应能导入） ==="
"$PY" - <<'PYEOF'
import os, sys
sys.path.insert(0, "/root/rivermind-data/vlx-seek")
for tag in ("sdpa", "eager"):
    os.environ["VLX_ATTN"] = tag
    for m in list(sys.modules):
        if "vlx_seek" in m:
            del sys.modules[m]
    try:
        from vlx_seek.models.vlx_seek_1_5.builder import load_pretrained_model  # noqa
        print(f"  VLX_ATTN={tag} → 导入 OK")
    except Exception as e:
        print(f"  VLX_ATTN={tag} → 失败 {type(e).__name__}: {str(e)[:150]}")
PYEOF

echo; echo "=== 3. flash_attn 编译前提 ==="
for t in nvcc gcc g++ ninja; do
  if command -v $t >/dev/null 2>&1; then
    echo "  ● $t  $(command -v $t)"
  else
    echo "  ○ 无 $t"
  fi
done
command -v nvcc >/dev/null 2>&1 && (nvcc --version | tail -2 | sed 's/^/  /')
echo "  CPU 核数 $(nproc)"
echo "  CUDA_HOME=${CUDA_HOME:-未设}"
ls -d /usr/local/cuda* 2>/dev/null | sed 's/^/  /'

echo; echo "=== 4. 是否存在匹配的预编译 wheel（torch2.10 cu128 cp310） ==="
curl -sL --max-time 60 \
  "https://api.github.com/repos/Dao-AILab/flash-attention/releases?per_page=8" \
  2>/dev/null | "$PY" - <<'PYEOF'
import json, sys
try:
    rels = json.load(sys.stdin)
except Exception as e:
    print(f"  取发布列表失败: {type(e).__name__}"); raise SystemExit
for r in rels if isinstance(rels, list) else []:
    names = [a["name"] for a in r.get("assets", [])]
    hit = [n for n in names if "cp310" in n and ("cu12" in n or "cu128" in n)]
    print(f"  {r.get('tag_name')}  资产 {len(names)} 个；cp310+cu12 命中 {len(hit)}")
    for n in hit[:4]:
        print(f"      {n}")
PYEOF

echo; echo "=== 5. GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | sed 's/^/  /'
echo "=== 结束 ==="

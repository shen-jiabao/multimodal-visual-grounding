#!/bin/bash
# VLX-Seek 准备 1：腾空间 + 取官方仓库 + 读依赖（不含模型权重下载）
set -u
R=/root/rivermind-data
M=$R/models
VLX=$R/vlx-seek

echo "=== 0. 删除前留痕 ==="
du -sh "$M/Qwen3-VL-235B-A22B-Instruct-AWQ" 2>/dev/null | sed 's/^/  将删 /'
ls -1 "$M/Qwen3-VL-235B-A22B-Instruct-AWQ" 2>/dev/null | wc -l | sed 's/^/  文件数 /'
rm -rf "$M/Qwen3-VL-235B-A22B-Instruct-AWQ"
echo "  已删除（用户明确授权）"
df -h "$R" | tail -1 | sed 's/^/  现磁盘 /'

echo; echo "=== 1. 取官方仓库 ==="
rm -rf "$VLX"; mkdir -p "$VLX"
ok=0
for url in "https://codeload.github.com/om-ai-lab/VLX-Seek/tar.gz/refs/heads/main" \
           "https://codeload.github.com/om-ai-lab/VLX-Seek/tar.gz/refs/heads/master"; do
  if curl -sSL --max-time 120 -o /tmp/vlx.tgz "$url" && [ -s /tmp/vlx.tgz ]; then
    tar -xzf /tmp/vlx.tgz -C "$VLX" --strip-components=1 && ok=1 && \
      echo "  [命中] $url" && break
  fi
done
[ "$ok" = 1 ] || { echo "  ★仓库下载失败"; exit 1; }
find "$VLX" -maxdepth 2 -type f | head -40 | sed "s|$VLX/|    |"

echo; echo "=== 2. requirements.txt ==="
cat "$VLX/requirements.txt" 2>/dev/null | sed 's/^/  /' || echo "  （无）"

echo; echo "=== 3. 官方 inference.md 关键段（接口/输出/参数） ==="
for f in "$VLX/docs/inference.md"; do
  [ -f "$f" ] && grep -inE "bbox-list|--task|output|json|obj[0-9]|region|detector|python .*infer|arg" "$f" \
    | head -40 | sed 's/^/  /'
done

echo; echo "=== 4. worker 的对外接口（类/方法签名） ==="
if [ -f "$VLX/vlx_seek_worker.py" ]; then
  grep -nE "^class |^    def |def __init__" "$VLX/vlx_seek_worker.py" | head -30 | sed 's/^/  /'
else
  find "$VLX" -name "*worker*.py" | sed 's/^/  找到 /'
fi

echo; echo "=== 5. 环境工具 ==="
"$R/venv/bin/python" -c "
import importlib.util as u
for m in ('huggingface_hub','modelscope'):
    print('  ', m, '有' if u.find_spec(m) else '无')"
ls "$R/venv/bin" | grep -iE "hf|huggingface" | sed 's/^/  cli /'
echo "=== 结束 ==="

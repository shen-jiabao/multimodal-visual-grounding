#!/bin/bash
# VLX-Seek 准备 1c：取全仓库源码文件（跳过 assets），并读真实接口
set -u
R=/root/rivermind-data
VLX=$R/vlx-seek
mkdir -p "$VLX"

echo "=== 拉取全部源码/配置文件 ==="
python3 - "$VLX" <<'PY'
import json, os, sys, urllib.request
VLX = sys.argv[1]
api = "https://api.github.com/repos/om-ai-lab/VLX-Seek/git/trees/main?recursive=1"
d = json.load(urllib.request.urlopen(api, timeout=60))
paths = [t["path"] for t in d.get("tree", []) if t["type"] == "blob"]
keep = [p for p in paths
        if p.endswith((".py", ".txt", ".md", ".json", ".jinja", ".yaml", ".yml"))
        and not p.startswith("assets/")]
ok = bad = 0
for p in keep:
    dst = os.path.join(VLX, p)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = f"https://raw.githubusercontent.com/om-ai-lab/VLX-Seek/main/{p}"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        if not data or data[:15] == b"404: Not Found":
            raise ValueError("empty")
        open(dst, "wb").write(data)
        ok += 1
    except Exception:
        bad += 1
        print("  ✗", p)
print(f"  取回 {ok} / {len(keep)}，失败 {bad}")
PY

echo; echo "=== 文件树（前 45） ==="
find "$VLX" -type f | sed "s|$VLX/||" | sort | head -45 | sed 's/^/  /'

echo; echo "=== inference.py 的 CLI 参数 ==="
grep -nE "add_argument|--[a-z-]+" "$VLX/inference.py" 2>/dev/null | head -30 | sed 's/^/  /'

echo; echo "=== worker: predict/ground/detect 签名 ==="
grep -nA6 -E "^    def (predict|ground|detect|count|run_task)\(" "$VLX/vlx_seek_worker.py" 2>/dev/null \
  | head -60 | sed 's/^/  /'

echo; echo "=== worker: 候选框相关 ==="
grep -nE "bbox_list|bbox-list|detector|we_detect|WeDetect|proposal" "$VLX/vlx_seek_worker.py" 2>/dev/null \
  | head -25 | sed 's/^/  /'

echo; echo "=== detect_tools/WeDetect ==="
find "$VLX/detect_tools" -type f 2>/dev/null | sed "s|$VLX/||" | sed 's/^/  /'
echo "=== 结束 ==="

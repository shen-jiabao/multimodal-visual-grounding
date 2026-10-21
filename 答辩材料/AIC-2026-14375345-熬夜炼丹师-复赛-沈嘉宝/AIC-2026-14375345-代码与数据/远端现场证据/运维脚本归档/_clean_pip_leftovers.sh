#!/bin/bash
# 只清 pip 遗留（用户已确认）：$R/tmp/pip-* 与 $R/pipcache。uvcache 保留。
set -u
R=/root/rivermind-data

echo "=== 清理前 ==="
du -sh "$R/tmp" "$R/pipcache" 2>/dev/null | sed 's/^/  /'
df -h "$R" | tail -1 | sed 's/^/  /'

echo
echo "=== 待删清单（只列 pip 遗留）==="
ls -1d "$R"/tmp/pip-* 2>/dev/null | sed 's/^/  /' || echo "  （无 pip-* 目录）"

echo
echo "=== 执行 ==="
n=$(ls -1d "$R"/tmp/pip-* 2>/dev/null | wc -l)
rm -rf "$R"/tmp/pip-* "$R/pipcache"
echo "  已删 $n 个 pip 临时目录 + pipcache"

echo
echo "=== 清理后 ==="
du -sh "$R/tmp" 2>/dev/null | sed 's/^/  /'
echo "  pipcache: $(ls -d "$R/pipcache" 2>/dev/null && echo '仍存在' || echo '已删除')"
echo "  --- $R/tmp 剩余内容 ---"
ls -1 "$R/tmp" 2>/dev/null | sed 's/^/    /' || echo "    （空）"
echo "  --- uvcache 保留 ---"
du -sh "$R/uvcache" 2>/dev/null | sed 's/^/    /'
df -h "$R" | tail -1 | sed 's/^/  /'
echo "=== DONE ==="

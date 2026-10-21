#!/bin/bash
# 闸门 3 的探测能力实测：故意造一个**截断**的 PNG，看 verify() 能否发现。
# 背景：项目踩过的坑是 `Image.load()` 抛 "image file is truncated"；
# 而闸门用的是 `Image.verify()`，二者检测范围不同，必须实测而非假定。
PY=/root/miniconda3/envs/myconda/bin/python
SRC=$(ls /srv/dataset/Images/visible/*.png | head -1)
echo "源图: $SRC  ($(stat -c%s "$SRC") 字节)"
mkdir -p /tmp/gatetest
$PY - <<PYEOF
import shutil, os
src = "$SRC"
dst = "/tmp/gatetest/trunc.png"
n = os.path.getsize(src)
with open(src, "rb") as f:
    data = f.read()
with open(dst, "wb") as f:
    f.write(data[: int(n * 0.6)])     # 截到 60%
print(f"截断副本: {dst}  {os.path.getsize(dst)} 字节 (原 {n})")
PYEOF
echo
echo '--- 用 verify() 探测截断副本 ---'
$PY - <<'PYEOF'
from PIL import Image
try:
    with Image.open("/tmp/gatetest/trunc.png") as im:
        im.verify()
    print("  verify(): 通过  <-- 闸门会漏判")
except Exception as e:
    print(f"  verify(): 抛错 {type(e).__name__}: {e}")
PYEOF
echo
echo '--- 用 load() 探测截断副本 ---'
$PY - <<'PYEOF'
from PIL import Image
try:
    with Image.open("/tmp/gatetest/trunc.png") as im:
        im.load()
    print("  load(): 通过")
except Exception as e:
    print(f"  load(): 抛错 {type(e).__name__}: {e}  <-- 真实失败模式")
PYEOF
echo
echo '--- 对照：完整原图 ---'
$PY - <<PYEOF
from PIL import Image
p = "$SRC"
for meth in ("verify", "load"):
    try:
        with Image.open(p) as im:
            getattr(im, meth)()
        print(f"  {meth}(): 通过")
    except Exception as e:
        print(f"  {meth}(): 抛错 {type(e).__name__}: {e}")
PYEOF
rm -rf /tmp/gatetest

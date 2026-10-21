#!/bin/bash
# 新机 bootstrap 第 1 步：打通 pip → 建 venv（放大盘）→ 装最小客户端依赖。
#
# 硬约束（实测）：系统盘 / 只有 30 GB，**任何包都不许装到 / 上**；
# 大盘为 /data/tini（900 GB 可用）与 /root/rivermind-data（49 GB 可用）。
# 本步只装 pillow + openai（api32b 路线的全部客户端依赖，~20 MB）。
set -u

RIVER=/root/rivermind-data
TINI=/data/tini
VENV=$RIVER/venv

echo "=== 0) 大盘可写性与余量 ==="
for d in "$RIVER" "$TINI"; do
  if [ -d "$d" ]; then
    if touch "$d/.wtest" 2>/dev/null; then rm -f "$d/.wtest"; w="可写"; else w="★不可写"; fi
    echo "  $d  $w  $(df -h "$d" | tail -1 | awk '{print $2\" 总, \"$4\" 可用\"}')"
  else
    echo "  $d 不存在"
  fi
done

echo
echo "=== 1) pip 是否可用 ==="
if python3 -m pip --version >/dev/null 2>&1; then
  echo "  已有 pip: $(python3 -m pip --version)"
else
  echo "  无 pip，尝试 ensurepip"
  if python3 -m ensurepip --upgrade >/dev/null 2>&1; then
    echo "  ensurepip 成功"
  else
    echo "  ensurepip 不可用，尝试 apt 安装 python3-pip python3-venv"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq >/dev/null 2>&1
    apt-get install -y -qq python3-pip python3-venv >/dev/null 2>&1
    if python3 -m pip --version >/dev/null 2>&1; then
      echo "  apt 安装成功: $(python3 -m pip --version)"
    else
      echo "  apt 也不行，改用 get-pip.py"
      curl -sSL --max-time 120 https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
        && python3 /tmp/get-pip.py --break-system-packages >/dev/null 2>&1
    fi
  fi
fi
python3 -m pip --version 2>&1 | head -1

echo
echo "=== 2) 建 venv 于大盘 ($VENV) ==="
if [ -x "$VENV/bin/python" ]; then
  echo "  已存在，跳过"
else
  python3 -m venv "$VENV" 2>&1 | tail -3
fi
"$VENV/bin/python" -V 2>&1
echo "  venv 体量: $(du -sh "$VENV" 2>/dev/null | cut -f1)"

echo
echo "=== 3) 装最小客户端依赖（pillow + openai）==="
"$VENV/bin/pip" install -q --upgrade pip 2>&1 | tail -2
"$VENV/bin/pip" install -q pillow openai 2>&1 | tail -5
echo "  --- 结果 ---"
"$VENV/bin/python" - <<'PYEOF'
import importlib.util as u
for n in ("PIL", "openai"):
    s = u.find_spec(n)
    if not s:
        print("  " + n + " MISSING")
    else:
        m = __import__(n)
        print("  " + n + " " + str(getattr(m, "__version__", "?")))
PYEOF

echo
echo "=== 4) 系统盘余量（必须几乎没变）==="
df -h / | tail -1
echo "  venv 与大文件都在大盘，系统盘只应增加极少"
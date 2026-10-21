#!/bin/bash
# 新机第二轮诊断：把第一轮暴露的四个问题逐一定位。
#   ① nvidia-smi 权限被拒 ⇒ 是无卡模式？还是设备节点缺失？
#   ② nvcc 不在 PATH ⇒ 但 /usr/local/cuda-13.0 存在，找它的 bin
#   ③ python3 找不到 ⇒ conda 未激活
#   ④ HuggingFace 直连不通 ⇒ 试国内镜像与 AutoDL 学术加速
set -u

echo "=== ① GPU 与设备节点 ==="
ls -l /usr/bin/nvidia-smi 2>&1 | sed 's/^/  /'
ls -l /dev/nvidia* 2>&1 | sed 's/^/  /'
echo "  --- 内核模块 ---"
lsmod 2>/dev/null | grep -i nvidia | head -3 | sed 's/^/  /' || echo "  （无 nvidia 模块信息）"
echo "  --- 直接试执行 ---"
/usr/bin/nvidia-smi 2>&1 | head -3 | sed 's/^/  /'
echo "  --- 退出码 $? ---"

echo
echo "=== ② nvcc 真实路径 ==="
for p in /usr/local/cuda/bin/nvcc /usr/local/cuda-13.0/bin/nvcc \
         /usr/local/cuda-13/bin/nvcc; do
  if [ -x "$p" ]; then echo "  ✓ $p"; "$p" --version | tail -2 | sed 's/^/      /'; fi
done
ls /usr/local/cuda-13.0/bin 2>/dev/null | head -20 | sed 's/^/  bin: /'

echo
echo "=== ③ Python / conda ==="
for p in /root/miniconda3/bin/python3 /root/miniconda3/envs/*/bin/python3; do
  [ -x "$p" ] && { echo "  ✓ $p → $("$p" -V 2>&1)"; }
done
/root/miniconda3/bin/conda env list 2>&1 | sed 's/^/  /'
echo "  conda 环境变量文件："
ls /etc/profile.d/conda.sh /root/miniconda3/etc/profile.d/conda.sh 2>&1 | sed 's/^/    /'

echo
echo "=== ④ 网络：直连 vs 国内镜像 vs AutoDL 学术加速 ==="
probe() {  # $1=标签 $2=url
  local code t
  t=$(curl -s -o /dev/null -w '%{http_code} %{time_total}' --max-time 12 "$2" 2>&1)
  echo "  $1  $t"
}
probe "pypi-直连      " https://pypi.org/simple/
probe "阿里pypi镜像   " https://mirrors.aliyun.com/pypi/simple/
probe "hf-直连        " https://huggingface.co
probe "hf-mirror镜像  " https://hf-mirror.com
probe "modelscope镜像 " https://www.modelscope.cn

if [ -f /etc/network_turbo ]; then
  echo "  --- 加载 /etc/network_turbo（AutoDL 学术加速）后重测 ---"
  # shellcheck disable=SC1091
  . /etc/network_turbo 2>/dev/null || true
  probe "hf-加速后      " https://huggingface.co
  probe "pypi-加速后    " https://pypi.org/simple/
else
  echo "  （无 /etc/network_turbo）"
fi

echo
echo "=== ⑤ 是否 AutoDL 无卡模式（只看得到卡才算有卡）==="
if [ -e /dev/nvidia0 ]; then echo "  有 /dev/nvidia0 ⇒ 已挂载 GPU"; else echo "  ★ 无 /dev/nvidia0 ⇒ 疑似无卡模式开机"; fi

echo
echo "=== DONE ==="

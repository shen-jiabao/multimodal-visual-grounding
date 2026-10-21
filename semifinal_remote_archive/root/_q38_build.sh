#!/bin/bash
# Qwen3.8-27B 环境构建（**独立 venv，不动现有 venv**）
#   依据（已核实）：
#     · vLLM 从 **v0.22.1** 起登记 `Qwen3_5ForConditionalGeneration`（v0.21.0 无）
#     · transformers 4.57.6 **不认识** model_type=qwen3_5 ⇒ 必须随新 vLLM 一起升级
#     · flash-linear-attention 有纯 python wheel（免编译）✓
#     · causal-conv1d **无 wheel** ⇒ 需 nvcc，本机无 ⇒ 待验证是否真被需要
set -u
R=/root/rivermind-data
# ★统一环境（PATH 含 cuda/bin 与 conda；缓存落数据盘；HF 走国内镜像）。
#   新机的 python 在 conda 里、nvcc 不在 PATH，且 `bash -s` 不加载 ~/.bashrc，
#   所以必须显式 source。旧平台无此文件时静默跳过，行为逐字不变。
[ -f "$R/_q38_env.sh" ] && . "$R/_q38_env.sh"
V=$R/venv_q38
LOG=$R/q38_env.log
VER=${Q38_VLLM:-0.23.0}

echo "=== 0) 磁盘（构建前） ===" > "$LOG"
df -h $R | tail -1 >> "$LOG" 2>&1

echo "=== 1) 建独立 venv（--clear 保证干净） ===" | tee -a "$LOG"
python3 -m venv --clear "$V" >> "$LOG" 2>&1 && echo "  ✓ $V" | tee -a "$LOG"
"$V/bin/python" -V 2>&1 | tee -a "$LOG"

echo "=== 2) 升级 pip（有 uv 则跳过——uv 不用 pip，省掉一次慢下载） ===" | tee -a "$LOG"
if [ -x /root/miniconda3/bin/uv ]; then
  echo "  ✓ 已装 uv，跳过 pip 升级" | tee -a "$LOG"
else
  "$V/bin/python" -m pip install -q -U pip >> "$LOG" 2>&1 && echo "  ✓" | tee -a "$LOG"
fi

echo "=== 3) 装 vLLM==$VER（含 torch；约 2-3 GB） ===" | tee -a "$LOG"
# ★装包器选择（实测依据）：本机 pip 只有 0.47 MB/s，而同一镜像 curl 达 16.8 MB/s、
#   盘写入 16.5 MB/s ⇒ 瓶颈在 pip 自身（逐个小请求串行等 TTFB），不是网络/磁盘/代理/IPv6。
#   uv 并发下载可绕开，故 uv 优先；无 uv 时回退 pip（行为与旧版一致）。
UVB=/root/miniconda3/bin/uv
IDX=${Q38_INDEX:-https://mirrors.aliyun.com/pypi/simple}
if [ -x "$UVB" ]; then
  echo "  装包器: uv $("$UVB" --version 2>&1 | awk '{print $2}')  index=$IDX" | tee -a "$LOG"
  UV_CACHE_DIR="$R/uvcache" UV_LINK_MODE=copy UV_HTTP_TIMEOUT=180 UV_PYTHON_DOWNLOADS=never \
    "$UVB" pip install --python "$V/bin/python" --default-index "$IDX" "vllm==$VER" >> "$LOG" 2>&1
else
  echo "  装包器: pip（未找到 uv）" | tee -a "$LOG"
  "$V/bin/python" -m pip install "vllm==$VER" >> "$LOG" 2>&1
fi
rc=$?
echo "  退出码 $rc" | tee -a "$LOG"
[ $rc -ne 0 ] && { echo "  ★vLLM 安装失败，日志尾 25 行：" | tee -a "$LOG"; tail -25 "$LOG"; }

echo "=== 4) 版本自检 ===" | tee -a "$LOG"
"$V/bin/python" - <<'PY' 2>&1 | tee -a "$LOG"
import importlib
for m in ("torch", "transformers", "vllm", "flash_linear_attention", "causal_conv1d", "triton"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:22s} {getattr(mod,'__version__','ok')}")
    except Exception as e:
        print(f"  {m:22s} ✗ {type(e).__name__}")
PY

echo "=== 5) ★关键：registry 是否登记 Qwen3_5ForConditionalGeneration ===" | tee -a "$LOG"
"$V/bin/python" - <<'PY' 2>&1 | tee -a "$LOG"
try:
    from vllm.model_executor.models.registry import ModelRegistry
    arch = sorted(ModelRegistry.get_supported_archs())
    hit = [a for a in arch if "Qwen3_5" in a]
    print(f"  架构总数 {len(arch)}")
    print(f"  Qwen3_5 系：{hit}")
    ok = "Qwen3_5ForConditionalGeneration" in arch
    print(f"  {'✓ 已登记（可加载 Qwen3.8-27B）' if ok else '✗ 未登记 ⇒ 此版本不可用'}")
except Exception as e:
    print(f"  ✗ 探测失败 {type(e).__name__}: {str(e)[:120]}")
PY

echo "=== 6) CUDA 可用性（无卡时应为 False，属正常） ===" | tee -a "$LOG"
"$V/bin/python" -c "import torch;print('  torch',torch.__version__,'| cuda build',torch.version.cuda,'| avail',torch.cuda.is_available())" 2>&1 | tee -a "$LOG"

echo "=== 7) 磁盘（构建后） ===" | tee -a "$LOG"
df -h $R | tail -1 >> "$LOG" 2>&1
echo "=== 结束 ===" | tee -a "$LOG"
tail -50 "$LOG"
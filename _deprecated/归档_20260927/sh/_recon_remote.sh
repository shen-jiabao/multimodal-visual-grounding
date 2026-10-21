#!/bin/bash
# 远端侦察：一次性摸清实例的硬件、环境、模型、网络与数据现状。
# 面向**全新实例**（/srv 为空、无预装 conda）编写，全部探测项都做了缺失容错。
echo '===== 1) 主机与磁盘 ====='
hostname; uptime
echo '--- 磁盘 ---'
df -h / /root /srv 2>/dev/null | sort -u
echo '--- 内存 ---'
free -g 2>/dev/null | head -3

echo
echo '===== 2) GPU ====='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo '--- GPU 占用进程 ---'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo '--- 各 GPU 进程命令行 ---'
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  printf '  pid %s: ' "$p"; tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null; echo
done

echo
echo '===== 3) Python 环境 ====='
echo '--- 系统 python ---'
which -a python3 python pip3 2>/dev/null
python3 -V 2>&1
echo '--- conda ---'
ls -d /root/miniconda3 /opt/conda 2>/dev/null || echo '  无 conda'
ls /root/miniconda3/envs 2>/dev/null || true
echo '--- 系统 python 关键包 ---'
python3 - <<'PY'
import sys
print("  py", sys.version.split()[0], "|", sys.executable)
for m in ("torch", "transformers", "PIL", "accelerate", "vllm", "openai", "modelscope"):
    try:
        mod = __import__(m)
        print(f"  {m:12s}", getattr(mod, "__version__", "?"))
    except Exception as e:
        print(f"  {m:12s} MISSING")
try:
    import torch
    print("  cuda_available", torch.cuda.is_available(),
          "| devices", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("  cap", torch.cuda.get_device_capability(0),
              "| torch cuda", torch.version.cuda)
except Exception as e:
    print("  torch 探测失败:", type(e).__name__)
PY

echo
echo '===== 4) 已有模型 / 缓存 ====='
echo '--- /root 内容 ---'
ls -la /root 2>/dev/null | head -30
echo '--- 全盘搜模型目录（限深 4）---'
find / -maxdepth 4 -type d \( -iname '*Qwen*' -o -iname '*models*' -o -iname '*GLM*' -o -iname '*Intern*' \) 2>/dev/null | grep -v -E '^/(proc|sys)' | head -20
echo '--- HF / ModelScope 缓存 ---'
du -sh /root/.cache/huggingface 2>/dev/null || echo '  无 HF 缓存'
du -sh /root/.cache/modelscope 2>/dev/null || echo '  无 MS 缓存'
echo '--- vLLM API 是否在监听 ---'
curl -s -m 8 http://127.0.0.1:8000/v1/models | head -c 400
echo

echo
echo '===== 5) 网络出口 ====='
for u in https://pypi.org https://huggingface.co https://www.modelscope.cn https://pypi.tuna.tsinghua.edu.cn; do
  printf '  %-40s ' "$u"
  curl -s -m 10 -o /dev/null -w '%{http_code}\n' "$u" 2>/dev/null || echo FAIL
done
echo '--- pip 源 ---'
pip3 config list 2>/dev/null; cat /etc/pip.conf 2>/dev/null; cat /root/.pip/pip.conf 2>/dev/null

echo
echo '===== 6) 数据现状 ====='
echo '--- /srv ---'
ls -la /srv 2>/dev/null
echo '--- 是否已有 queries.json ---'
find / -maxdepth 4 -name 'queries.json' 2>/dev/null | grep -v -E '^/(proc|sys)' | head
echo '--- 远端已有输出 ---'
ls -la /srv/output 2>/dev/null || echo '  无 /srv/output'

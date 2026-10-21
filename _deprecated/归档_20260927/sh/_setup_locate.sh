#!/bin/bash
# LocateAnything-3B 装配：下载权重 + 建隔离 venv + 装依赖 + 自检。
# 配方取自初赛成功路径（_setup_px2.sh / _venv_fix.sh），但**不盲抄**：
# 初赛 venv 是 py3.13，而 `numpy<2` 在 py3.13 上无 wheel（numpy 2.1.0 才
# 支持 cp313）。故本脚本先试全量 pin，失败则去掉 numpy pin 重试，并**如实
# 打印两次的 RC 与最终版本**，绝不把失败包装成成功。
LOG=/srv/output/setup_locate.log
exec > >(tee -a $LOG) 2>&1
PY=/root/miniconda3/envs/myconda/bin/python
echo "=== [$(date)] 开始装配 LocateAnything-3B ==="

echo "== 0) 基线环境 =="
$PY -c "import sys, numpy; print('py', sys.version.split()[0]); print('numpy', numpy.__version__)" 2>&1

echo
echo "== 1) 下载模型（7.2G，ModelScope）=="
mkdir -p /srv/models
if [ -f /srv/models/LocateAnything-3B/model.safetensors.index.json ]; then
  echo "模型已存在，跳过下载"
else
  export MODELSCOPE_DOMAIN=www.modelscope.cn
  $PY - <<'PY'
from modelscope import snapshot_download
p = snapshot_download("nv-community/LocateAnything-3B",
                      local_dir="/srv/models/LocateAnything-3B")
print("模型目录:", p)
PY
fi
echo "-- 模型体积 --"
du -sh /srv/models/LocateAnything-3B 2>/dev/null || echo "(缺失)"
ls /srv/models/LocateAnything-3B 2>/dev/null | head -20

echo
echo "== 2) 建模隔离 venv（复用 myconda 的 torch）=="
rm -rf /srv/venv-locate
$PY -m venv --system-site-packages /srv/venv-locate
/srv/venv-locate/bin/python -c "import torch; print('venv 内 torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo
echo "== 3) 装依赖：先试全量 pin（含 numpy<2）=="
/srv/venv-locate/bin/pip install -q --index-url https://pypi.org/simple/ \
  "transformers==4.57.1" "tokenizers==0.22.0" "huggingface-hub>=0.36" \
  "accelerate>=1.5" "peft==0.12.0" "Pillow>=11" "numpy<2" \
  "opencv-python-headless==4.11.0.86" "decord==0.6.0" "lmdb==1.7.5" \
  2>/tmp/pip_full.log
RC_FULL=$?
echo "PIP_RC(全量 pin) = $RC_FULL"
tail -6 /tmp/pip_full.log

if [ "$RC_FULL" -ne 0 ]; then
  echo
  echo "== 3b) 全量 pin 失败，去掉 numpy pin 重试 =="
  /srv/venv-locate/bin/pip install -q --index-url https://pypi.org/simple/ \
    "transformers==4.57.1" "tokenizers==0.22.0" "huggingface-hub>=0.36" \
    "accelerate>=1.5" "peft==0.12.0" "Pillow>=11" \
    2>/tmp/pip_nonp.log
  echo "PIP_RC(去 numpy pin) = $?"
  tail -6 /tmp/pip_nonp.log
fi

echo
echo "== 4) 自检 =="
/srv/venv-locate/bin/python - <<'PY'
for m in ("torch", "transformers", "tokenizers", "numpy", "PIL", "peft", "accelerate", "cv2", "decord", "lmdb"):
    try:
        mod = __import__(m)
        print(f"  {m:14s}", getattr(mod, "__version__", "?"))
    except Exception as e:
        print(f"  {m:14s} MISSING ({type(e).__name__})")
PY

echo
echo "== 5) 模型目录文件清单 =="
ls -l /srv/models/LocateAnything-3B 2>/dev/null | head -25
echo "=== [$(date)] 装配结束 ==="

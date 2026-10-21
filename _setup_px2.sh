#!/bin/bash
# px2 装配：LocateAnything 依赖 + 隔离 venv + 下载模型权重到 /srv/models
set -e
exec > >(tee -a /root/setup.log) 2>&1
echo "=== [$(date)] 开始装配 ==="

# 1. 下载模型（ModelScope，7.2G）
mkdir -p /srv/models
export MODELSCOPE_DOMAIN=www.modelscope.ai
if [ ! -f /srv/models/LocateAnything-3B/model.safetensors.index.json ]; then
  echo "=== 下载 LocateAnything-3B ==="
  python3 -m pip install -q modelscope 2>&1 | tail -2
  python3 -c "
from modelscope import snapshot_download
p = snapshot_download('nv-community/LocateAnything-3B', local_dir='/srv/models/LocateAnything-3B')
print('模型目录:', p)
" || {
  echo "modelscope 失败，尝试 mkdir+git 方式" ;
  mkdir -p /srv/models/LocateAnything-3B ;
  cd /srv/models/LocateAnything-3B ;
  git lfs install 2>/dev/null || true ;
  git clone https://www.modelscope.cn/nv-community/LocateAnything-3B.git /srv/models/LocateAnything-3B 2>&1 | tail -3 ;
}
else
  echo "模型已存在，跳过下载"
fi
du -sh /srv/models/LocateAnything-3B 2>/dev/null || true

# 2. 建隔离 venv（复用 torch 2.8，装 transformers 等）
echo "=== 建 venv ==="
python3 -m venv --system-site-packages /srv/venv-locate
/srv/venv-locate/bin/pip install -q \
  "transformers==4.57.1" \
  "tokenizers==0.22.0" \
  "huggingface-hub>=0.36" \
  "accelerate>=1.5" \
  "peft==0.12.0" \
  "Pillow>=11" \
  "numpy<2" \
  "opencv-python-headless==4.11.0.86" \
  "decord==0.6.0" \
  "lmdb==1.7.5" 2>&1 | tail -3

echo "=== 验证 ==="
/srv/venv-locate/bin/python -c "
import torch, transformers, PIL
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('PIL', PIL.__version__)
print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
"
echo "=== 磁盘 ==="
df -h /srv | tail -1
echo "=== [$(date)] 装配完成 ==="
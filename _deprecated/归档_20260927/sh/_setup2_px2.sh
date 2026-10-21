#!/bin/bash
# 补装其余依赖 + 重下模型实体
set -e
echo "=== 补依赖 ==="
/srv/venv-locate/bin/pip install -q \
  --index-url https://pypi.org/simple/ \
  "transformers==4.57.1" "tokenizers>=0.20" \
  "accelerate>=1.5" "peft>=0.12" "numpy<2" \
  "Pillow>=11" "opencv-python-headless==4.11.0.86" \
  "decord==0.6.0" "lmdb==1.7.5" 2>/tmp/pip_err3.log
echo "PIP3_RC=$?"
/srv/venv-locate/bin/python - <<'PY'
import torch, transformers, PIL
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("tf", transformers.__version__)
print("PIL", PIL.__version__)
PY

echo "=== 重下模型（modelscope） ==="
rm -rf /srv/models/LocateAnything-3B
export MODELSCOPE_DOMAIN=www.modelscope.cn
python3 -m pip install -q modelscope 2>/dev/null || true
python3 -c "
from modelscope import snapshot_download
p = snapshot_download('nv-community/LocateAnything-3B', local_dir='/srv/models/LocateAnything-3B')
print('模型目录:', p)
" 2>&1 | tail -3
echo "MODEL_SIZE:"
du -sh /srv/models/LocateAnything-3B 2>/dev/null || echo "(未下载成功)"
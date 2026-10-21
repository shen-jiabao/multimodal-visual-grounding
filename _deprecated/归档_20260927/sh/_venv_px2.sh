#!/bin/bash
# 安装 Locate 依赖到隔离 venv
set -e
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
  "lmdb==1.7.5" 2>/tmp/pip_err.log
echo "PIP_RC=$?"
/srv/venv-locate/bin/python - <<'PY'
try:
    import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())
except Exception as ex: print("torch ERR", ex)
import transformers; print("transformers", transformers.__version__)
import PIL; print("PIL", PIL.__version__)
PY
tail -3 /tmp/pip_err.log 2>/dev/null || true
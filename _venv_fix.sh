#!/bin/bash
# 用官方 PyPI 源重装依赖到隔离 venv（py3.13）
set -x
/srv/venv-locate/bin/pip install -q \
  --index-url https://pypi.org/simple/ \
  "transformers==4.57.1" \
  "tokenizers>=0.20" \
  "huggingface-hub>=0.36" \
  "accelerate>=1.5" \
  "peft>=0.12" \
  "numpy<2" \
  2>/tmp/pip_err2.log
echo "PIP2_RC=$?"
/srv/venv-locate/bin/python -c "import transformers; print('tf', transformers.__version__)" 2>&1 | tail -1
tail -5 /tmp/pip_err2.log
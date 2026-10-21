#!/bin/bash
# 诊断 /srv/venv-locate 为何 import transformers.AutoProcessor 失败。
# 症状：ImportError: cannot import name 'AutoProcessor'（4.57.1 本应有）；
#       且 peft 也是 ImportError 而非 ModuleNotFoundError。
PY=/root/miniconda3/envs/myconda/bin/python
VPY=/srv/venv-locate/bin/python

echo "===== A) 版本与路径 ====="
$VPY -c "import transformers, sys; print('ver', transformers.__version__); print('file', transformers.__file__); print('py', sys.version.split()[0]); print('sys.path[:4]', sys.path[:4])" 2>&1 | tail -8

echo
echo "===== B) 逐个 import 定位 ====="
for n in AutoModel AutoTokenizer AutoProcessor AutoConfig AutoImageProcessor; do
  $VPY -c "from transformers import $n; print('  OK   $n')" 2>&1 | tail -2
done

echo
echo "===== C) 直接看 __init__ 是否完整 ====="
$VPY - <<'PY' 2>&1 | tail -12
import os, transformers
d = os.path.dirname(transformers.__file__)
print("has __init__.py:", os.path.exists(os.path.join(d, "__init__.py")))
print("has AutoProcessor source:",
      os.path.exists(os.path.join(d, "models", "auto", "processing_auto.py")))
print("has AutoModel source:",
      os.path.exists(os.path.join(d, "models", "auto", "modeling_auto.py")))
print("dist-info 目录:")
sd = os.path.dirname(d)
for n in sorted(os.listdir(sd)):
    if "transformers" in n or "tokenizers" in n or "huggingface" in n:
        print("   ", n)
PY

echo
echo "===== D) pip check / 关键包 ====="
/srv/venv-locate/bin/pip check 2>&1 | head -12
echo "--- pip list ---"
/srv/venv-locate/bin/pip list 2>/dev/null | grep -iE 'transformers|tokenizers|huggingface|numpy|peft|accelerate|torch' 

echo
echo "===== E) myconda 侧对照（它能 import AutoProcessor 吗）====="
$PY -c "import transformers; print('myconda transformers', transformers.__version__); from transformers import AutoProcessor; print('myconda AutoProcessor OK')" 2>&1 | tail -4

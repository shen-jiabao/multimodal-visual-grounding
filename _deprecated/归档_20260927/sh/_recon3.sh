PY=/root/miniconda3/envs/myconda/bin/python
echo ====MYCONDA_VER====
$PY -V
echo ====MYCONDA_DEPS====
$PY - <<'PYEOF'
import importlib
for m in ["torch","torchvision","transformers","numpy","cv2","PIL","timm","yaml","einops","scipy"]:
    try:
        mod=importlib.import_module(m)
        print(m, getattr(mod,"__version__","ok"))
    except Exception as e:
        print(m,"MISSING",type(e).__name__,str(e)[:80])
try:
    import torch
    print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
    print("torch_cuda_ver", torch.version.cuda)
except Exception as e:
    print("torch_err", e)
PYEOF
echo ====BUILDTOOLS====
which gcc g++ nvcc 2>/dev/null; ls /usr/local/cuda*/bin/nvcc 2>/dev/null; echo "CUDA_HOME=$CUDA_HOME"
echo ====PIPLIST====
$PY -m pip list 2>/dev/null | grep -iE '^(torch|torchvision|transformers|opencv|numpy|pillow|timm|einops) ' 
echo ====MEM====
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

PY=/root/miniconda3/envs/myconda/bin/python
export HF_ENDPOINT=https://hf-mirror.com

echo "===== A) 试阿里云源装 open_clip_torch（可选） ====="
timeout 180 $PY -m pip install -q -i https://mirrors.aliyun.com/pypi/simple/ open_clip_torch 2>&1 | tail -3
$PY -c 'import open_clip; print("open_clip OK", open_clip.__version__)' 2>&1 | tail -1

echo "===== B) transformers CLIP 权重下载（hf-mirror）+ CPU 自测 ====="
$PY - <<'PYEOF'
import os, time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "0"
t0 = time.time()
from transformers import CLIPModel, CLIPProcessor
name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(name)
proc = CLIPProcessor.from_pretrained(name)
print(f"CLIP 加载成功，用时 {time.time()-t0:.1f}s")
model.eval()
import torch, numpy as np
from PIL import Image
im = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype("uint8"))
inp = proc(text=["a red promotional sign", "a blue curtain"], images=im,
           return_tensors="pt", padding=True)
with torch.no_grad():
    out = model(**inp)
print("logits_per_image:", out.logits_per_image.detach().tolist())
PYEOF
echo "===== C) 缓存占用 ====="
du -sh /root/.cache/huggingface 2>/dev/null

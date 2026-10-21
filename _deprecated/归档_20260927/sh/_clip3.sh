PY=/root/miniconda3/envs/myconda/bin/python
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_OFFLINE=0

echo "===== open_clip 加载 ViT-B-32（openai -> laion 回退），CPU 打分自测 ====="
$PY - <<'PYEOF'
import torch, open_clip, time
torch.set_num_threads(4)
def try_load(name):
    t0=time.time()
    m,_,pre = open_clip.create_model_and_transforms('ViT-B-32', pretrained=name)
    tok = open_clip.get_tokenizer('ViT-B-32')
    m.eval()
    img = torch.rand(1,3,224,224)
    with torch.no_grad():
        f=m.encode_image(img); t=m.encode_text(tok(['a red sign','blue curtain']))
        f=f/f.norm(dim=-1,keepdim=True); t=t/t.norm(dim=-1,keepdim=True)
    print(f"OK pretrained={name} 用时{time.time()-t0:.0f}s sim={(f@t.T).tolist()}")
    return True
for name in ['openai','laion2b_s34b_b79k','laion400m_e32']:
    try:
        try_load(name); break
    except Exception as e:
        print(f"失败 {name}: {type(e).__name__} {str(e)[:200]}")
PYEOF

echo "===== 若 open_clip 全失败：用 snapshot_download(禁用xet) 拉 transformers CLIP ====="
$PY - <<'PYEOF'
import os, time
os.environ.update(dict(HF_ENDPOINT="https://hf-mirror.com", HF_HUB_DISABLE_XET="1"))
try:
    import open_clip; print("open_clip 可用，跳过"); raise SystemExit
except SystemExit: raise
except Exception: pass
from huggingface_hub import snapshot_download
t0=time.time()
p = snapshot_download("openai/clip-vit-base-patch32",
                      allow_patterns=["*.json","*.txt","model.safetensors"])
print("下载完成", round(time.time()-t0), "s ->", p)
from transformers import CLIPModel, CLIPProcessor
m = CLIPModel.from_pretrained(p); pr = CLIPProcessor.from_pretrained(p); m.eval()
import torch, numpy as np
from PIL import Image
im = Image.fromarray((np.random.rand(256,256,3)*255).astype("uint8"))
inp = pr(text=["a red sign","blue curtain"], images=im, return_tensors="pt", padding=True)
with torch.no_grad(): out = m(**inp)
print("transformers CLIP OK, logits=", out.logits_per_image.detach().tolist())
PYEOF
echo "===== 缓存占用 ====="
du -sh /root/.cache/huggingface 2>/dev/null

PY=/root/miniconda3/envs/myconda/bin/python
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "===== 1) 安装 open_clip_torch (清华镜像) ====="
$PY -m pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple open_clip_torch 2>&1 | tail -5
$PY -c 'import open_clip; print("open_clip", open_clip.__version__)'

echo "===== 2) 下载/加载 ViT-B-32 (openai) 并做 CPU 打分自测 ====="
$PY - <<'PYEOF'
import torch, open_clip
try:
    m, _, pre = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tok = open_clip.get_tokenizer('ViT-B-32')
    m.eval()
    img = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        f = m.encode_image(img)
        t = m.encode_text(tok(['a red promotional sign', 'blue curtain']))
        f = f / f.norm(dim=-1, keepdim=True)
        t = t / t.norm(dim=-1, keepdim=True)
        print('OK open_clip ViT-B-32; sim=', (f @ t.T).tolist())
except Exception as e:
    print('open_clip 失败:', type(e).__name__, str(e)[:300])
    print('--- 回退 transformers CLIP ---')
    from transformers import CLIPModel, CLIPProcessor
    cm = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    cp = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    cm.eval()
    from PIL import Image
    import numpy as np
    im = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype('uint8'))
    inp = cp(text=['a red sign'], images=im, return_tensors='pt', padding=True)
    with torch.no_grad():
        out = cm(**inp)
    print('OK transformers CLIP; logits_per_image=',
          out.logits_per_image.detach().tolist())
PYEOF
echo "===== 3) 缓存占用 ====="
du -sh /root/.cache/huggingface 2>/dev/null

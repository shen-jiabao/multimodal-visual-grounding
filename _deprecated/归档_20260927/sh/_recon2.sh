echo ====MINICONDA====
ls /root/miniconda3/envs 2>/dev/null
/root/miniconda3/bin/python -V
echo ====PYDEPS====
/root/miniconda3/bin/python - <<'PYEOF'
import importlib
for m in ["torch","torchvision","transformers","numpy","cv2","PIL","timm","yaml","supervision"]:
    try:
        mod=importlib.import_module(m)
        print(m, getattr(mod,"__version__","ok"))
    except Exception as e:
        print(m, "MISSING", e)
try:
    import torch
    print("cuda_available", torch.cuda.is_available(), "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
except Exception as e:
    print("torch import err", e)
PYEOF
echo ====GDFIND====
find /root /opt /usr/local -maxdepth 6 \( -type d -name 'groundingdino' -o -name 'groundingdino_swint_ogc.pth' -o -name 'slconfig.py' \) 2>/dev/null | head -20
echo ====QJ====
/root/miniconda3/bin/python - <<'PYEOF'
import json, os
q=json.load(open('/root/dataset/初赛数据集-基于大模型的多模态视觉理解与推理/queries/queries.json'))
ks=list(q.keys()); print('qids',len(ks),'first',ks[0])
k=ks[0]; rec=q[k]; print('fields', list(rec.keys()))
print(json.dumps(rec, ensure_ascii=False)[:300])
print('has 000282_004', '000282_004' in q)
PYEOF
echo ====IMAGES====
ls '/root/dataset/初赛数据集-基于大模型的多模态视觉理解与推理/Images/visible' | wc -l
ls '/root/dataset/初赛数据集-基于大模型的多模态视觉理解与推理/Images/visible/000282.png' 2>&1
echo ====PS====
ps aux | grep -E 'python|PID' | grep -v grep | head -10
echo ====DISK====
df -h /root | tail -1

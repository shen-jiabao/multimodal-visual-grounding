echo "=== HF 连通性 ==="
curl -sI --max-time 10 https://huggingface.co | head -3
echo "=== 配置改回 bert-base-uncased ==="
sed -i "s#text_encoder_type = '.*'#text_encoder_type = 'bert-base-uncased'#" /root/infer/GD/groundingdino/config/GroundingDINO_SwinT_OGC.py
grep -n 'text_encoder_type =' /root/infer/GD/groundingdino/config/GroundingDINO_SwinT_OGC.py
echo "=== 冒烟重跑 ==="
cd /root/infer
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 TOKENIZERS_PARALLELISM=false CUDA_DEVICE_ORDER=PCI_BUS_ID
time /root/miniconda3/envs/myconda/bin/python -u predict_gd_3modal.py --n 2 --out /root/infer/smoke2.json 2>&1 | grep -vE 'Warning|warn|UserWarning|FutureWarning|timm|meshgrid|use_return_dict|checkpoint:|autocast|^\s+return|^\s+warnings|^\s+with' | head -40
echo "=== output ==="
head -c 1200 /root/infer/smoke2.json
echo

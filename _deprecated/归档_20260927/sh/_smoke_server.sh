/root/miniconda3/envs/myconda/bin/python -c 'import pycocotools; print("pycocotools ok")'
cd /root/infer
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 TOKENIZERS_PARALLELISM=false CUDA_DEVICE_ORDER=PCI_BUS_ID
time /root/miniconda3/envs/myconda/bin/python -u predict_gd_3modal.py --n 2 --out /root/infer/smoke.json
echo "=== smoke output ==="
head -c 2500 /root/infer/smoke.json
echo

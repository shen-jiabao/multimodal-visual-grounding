sed -i "s#text_encoder_type = '.*'#text_encoder_type = '/root/infer/bert'#" /root/infer/GD/groundingdino/config/GroundingDINO_SwinT_OGC.py
grep -n 'text_encoder_type =' /root/infer/GD/groundingdino/config/GroundingDINO_SwinT_OGC.py
cd /root/infer
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_DEVICE_ORDER=PCI_BUS_ID
time /root/miniconda3/envs/myconda/bin/python -u predict_gd_3modal.py --n 3 --out /root/infer/smoke3.json > /root/infer/smoke3.log 2>&1
echo "rc=$?"
echo "=== 是否出现随机初始化告警 ==="
grep -c "were not initialized" /root/infer/smoke3.log || true
echo "=== 关键日志 ==="
grep -E "final text_encoder_type|模型就绪|共 |\[3/3\]|结果已保存" /root/infer/smoke3.log
echo "=== 输出片段 ==="
head -c 1500 /root/infer/smoke3.json
echo

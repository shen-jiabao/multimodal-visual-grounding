cd /root/infer
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_DEVICE_ORDER=PCI_BUS_ID
setsid nohup nice -n 10 /root/miniconda3/envs/myconda/bin/python -u predict_gd_3modal.py \
    --out /root/infer/result_gd_3modal_v2.json \
    > /root/infer/run.log 2>&1 < /dev/null &
sleep 4
echo "PID=$(pgrep -f 'predict_gd_3modal.py' | head -1)"
echo "started"

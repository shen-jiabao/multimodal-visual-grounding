echo "########## A) /models 与模型实际权重 ##########"
ls -la /models 2>/dev/null || echo "/models 不存在"
du -sh /models/* 2>/dev/null
echo "--- LocateAnything 权重 ---"
find /models /root/LocateAnything-3B -maxdepth 3 -name '*.safetensors' -o -maxdepth 3 -name '*.bin' 2>/dev/null | head
echo "--- VLX-Seek 权重 ---"
find /root/VLX-Seek -maxdepth 3 -name '*.safetensors' 2>/dev/null | head -5
du -sh /root/VLX-Seek/VLX-Seek-1.5-10B /root/VLX-Seek/WeDetect 2>/dev/null

echo "########## B) 历史预测文件完整性 ##########"
for f in /root/output/*.json; do
  echo "---- $f ----"
  /root/miniconda3/envs/myconda/bin/python - "$f" <<'PYEOF'
import json,sys
p=sys.argv[1]
d=json.load(open(p,encoding='utf-8'))
print("  条目:",len(d))
k=list(d)[0]; print("  首条:",k,json.dumps(d[k],ensure_ascii=False)[:220])
PYEOF
done

echo "########## C) v3 / enrich / vlxseek 脚本要点 ##########"
for f in /root/locate_run_v3.py /root/enrich_queries.py /root/vlxseek_infer.py; do
  if [ -f "$f" ]; then echo "===== $f ====="; head -45 "$f"; fi
done

echo "########## D) hf-mirror / 外网可达性 ##########"
curl -s -o /dev/null -w 'hf-mirror: %{http_code} %{time_total}s\n' --max-time 8 https://hf-mirror.com 2>&1
curl -s -o /dev/null -w 'modelscope: %{http_code} %{time_total}s\n' --max-time 8 https://modelscope.cn 2>&1
curl -s -o /dev/null -w 'pypi: %{http_code} %{time_total}s\n' --max-time 8 https://pypi.org/simple/ 2>&1

echo "########## E) 磁盘与 GPU 现状 ##########"
df -h / | tail -1
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

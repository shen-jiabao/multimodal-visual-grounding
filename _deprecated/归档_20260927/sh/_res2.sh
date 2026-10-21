echo "########## 1) 顶层模型/输出目录 ##########"
for d in /root/LocateAnything-3B /root/VLX-Seek /root/GroundingDINObase /root/output /root/infer /root/dataset; do
  if [ -e "$d" ]; then echo "== $d =="; du -sh "$d" 2>/dev/null; ls -la "$d" | head -20; else echo "== $d 不存在 =="; fi
done

echo "########## 2) output 目录细节（历史推理产物）##########"
find /root/output -maxdepth 2 -type f 2>/dev/null | head -20
find /root/output -maxdepth 2 -type f 2>/dev/null | wc -l

echo "########## 3) 现有推理脚本关键配置 ##########"
for f in /root/locate_run.py /root/locate_run_v2.py /root/vlxseek_infer.py /root/probe_api.py /root/retest_speed.py; do
  if [ -f "$f" ]; then echo "===== $f ($(wc -l < "$f") 行) ====="; head -40 "$f"; fi
done

echo "########## 4) conda 环境与关键依赖 ##########"
ls /root/miniconda3/envs
for py in /root/miniconda3/envs/*/bin/python; do
  echo "---- $py ----"
  $py -V 2>&1
  $py - <<'PYEOF' 2>&1 | tail -12
import importlib
for m in ["torch","transformers","open_clip","clip","cv2","numpy","PIL","timm","sentence_transformers","accelerate","peft"]:
    try:
        mod=importlib.import_module(m)
        print(" OK ", m, getattr(mod,"__version__","-"))
    except Exception as e:
        print(" -- ", m, "MISSING")
PYEOF
done

echo "########## 5) HF 本地缓存（可离线复用的模型）##########"
ls -d /root/.cache/huggingface/hub/models--* 2>/dev/null | sed 's#.*/models--##' | head -30
du -sh /root/.cache/huggingface 2>/dev/null

echo "########## 6) 残留训练/LoRA/融合相关脚本 ##########"
ls /root/*.py /root/*.sh 2>/dev/null
grep -rIl --include=*.py -e "lora" -e "LoRA" -e "clip" -e "ensemble" -e "fuse" /root --exclude-dir=miniconda3 --exclude-dir=dataset 2>/dev/null | head -20

echo "########## 7) 当前算力占用 ##########"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af 'vlxseek|locate' | grep -v pgrep | head

echo "########## 8) chunks 下载目录 ##########"
ls -la /root/chunks 2>/dev/null | head -15
du -sh /root/chunks 2>/dev/null

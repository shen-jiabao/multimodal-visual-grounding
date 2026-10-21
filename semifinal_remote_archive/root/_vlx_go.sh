#!/bin/bash
# VLX-Seek 准备 2：独立 venv + 依赖安装 + 权重下载（全部后台分离，不随 SSH 断）
set -u
R=/root/rivermind-data
VLX=$R/vlx-seek
VLXVENV=$R/venv_vlx
RES=$VLX/resources
mkdir -p "$RES"

echo "=== 磁盘 ==="
df -h "$R" | tail -1 | sed 's/^/  /'

echo; echo "=== A. 独立 venv（不与 vLLM 环境混） ==="
if [ ! -x "$VLXVENV/bin/python" ]; then
  python3 -m venv "$VLXVENV" && echo "  已建 $VLXVENV" || { echo "  ★建 venv 失败"; exit 1; }
else
  echo "  已存在 $VLXVENV"
fi

cat > /tmp/_vlx_pip.sh <<'EOS'
#!/bin/bash
R=/root/rivermind-data; VLXVENV=$R/venv_vlx; VLX=$R/vlx-seek
export PIP_INDEX_URL=https://pypi.org/simple
$VLXVENV/bin/pip install -U pip wheel setuptools
echo "---- 主依赖 ----"
$VLXVENV/bin/pip install \
  "torch==2.10.0" "torchvision==0.25.0" "transformers==5.13.0" \
  "timm==1.0.9" "einops==0.6.1" "accelerate==1.4.0" \
  pillow numpy safetensors huggingface_hub 2>&1 | tail -25
echo "---- flash_attn（可能需编译，失败不阻塞） ----"
$VLXVENV/bin/pip install "flash_attn==2.8.3" 2>&1 | tail -12 \
  || echo "[warn] flash_attn 未装上（快路径不可用，慢路径仍可跑）"
echo "---- 自检 ----"
$VLXVENV/bin/python -c "
import torch, transformers, timm
print('OK torch', torch.__version__, '| tf', transformers.__version__, '| timm', timm.__version__)
" 2>&1 | tail -3
echo "[pip] DONE"
EOS
chmod +x /tmp/_vlx_pip.sh
nohup bash /tmp/_vlx_pip.sh > "$R/vlx_pip.log" 2>&1 &
echo "  已后台启动 pip  PID=$!"

echo; echo "=== B. 下载权重 omlab/VLX-Seek-1.5-10B（约 20G，走 hf-mirror） ==="
cat > /tmp/_vlx_dl_model.sh <<EOS
#!/bin/bash
set -u
D=$RES/VLX-Seek-1.5-10B
mkdir -p "\$D"
B=https://hf-mirror.com/omlab/VLX-Seek-1.5-10B/resolve/main
for f in config.json generation_config.json preprocessor_config.json \\
         tokenizer.json tokenizer_config.json chat_template.jinja model.safetensors; do
  echo "[\$(date +%H:%M:%S)] \$f"
  curl -sSL --retry 5 --retry-delay 5 -C - -o "\$D/\$f" "\$B/\$f" || echo "  ✗ \$f"
done
echo "---- 结果 ----"
ls -la "\$D" | sed 's/^/  /'
du -sh "\$D"
echo "[dl_model] DONE"
EOS
nohup bash /tmp/_vlx_dl_model.sh > "$R/vlx_dl_model.log" 2>&1 &
echo "  已后台启动  PID=$!"

echo; echo "=== C. WeDetect 仓库文件清单 ==="
curl -sS --max-time 30 "https://hf-mirror.com/api/models/fushh7/WeDetect" 2>/dev/null \
  | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception as e: print('  查询失败', type(e).__name__); raise SystemExit
for s in d.get('siblings',[]): print('  ', s.get('rfilename'))
" 2>&1 | head -20

echo; echo "=== 启动确认 ==="
sleep 3
pgrep -af "pip|_vlx_dl_model" | head -5 | sed 's/^/  /'
echo "=== 结束（长任务已在后台跑，用 _vlx_prog.sh 查进度） ==="

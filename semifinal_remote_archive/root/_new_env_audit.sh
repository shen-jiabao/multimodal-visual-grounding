#!/bin/bash
# 新机环境审计：按用户 6 行要求逐条核验，并检查"是否干净（只有这一个模型）"
set -u
R=/root/rivermind-data
D=$R/models/Qwen3.8-27B
PY=$R/venv_q38/bin/python

echo "############ A) 模型身份：直接读权重头，不看目录名 ############"
$PY - "$D" <<'PY'
import json, os, struct, sys, collections
d = sys.argv[1]

cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
print("--- config.json 关键字段 ---")
for k in ("model_type", "architectures", "torch_dtype", "dtype",
          "num_hidden_layers", "hidden_size", "vocab_size",
          "num_attention_heads", "num_key_value_heads", "tie_word_embeddings"):
    if k in cfg:
        print(f"  {k:24s} = {cfg[k]}")
print(f"  quantization_config     = {cfg.get('quantization_config', '【不存在】✓ 未量化')}")
print(f"  （含 'quant' 字样的字段: {[k for k in cfg if 'quant' in k.lower()] or '无'}）")

idx = os.path.join(d, "model.safetensors.index.json")
wm = json.load(open(idx, encoding="utf-8"))["weight_map"]
shards = sorted(set(wm.values()))

# ★核心：逐分片读 safetensors 头，统计 dtype 与元素数（=真实参数量）
dtypes = collections.Counter()
total = 0
per = {}
for s in shards:
    p = os.path.join(d, s)
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    cnt = 0
    for name, info in hdr.items():
        if name == "__metadata__":
            continue
        dt = info["dtype"]
        numel = 1
        for x in info["shape"]:
            numel *= x
        dtypes[dt] += numel
        cnt += numel
    per[s] = cnt
    total += cnt

print("--- 权重实况（读 safetensors 头，非猜测）---")
print(f"  分片数              = {len(shards)}")
print(f"  参数总量            = {total/1e9:.3f} B  ({total:,} 个)")
print(f"  dtype 分布          = {dict(dtypes)}")
print(f"  是否全为 BF16       = {set(dtypes) == {'BF16'}}")
print(f"  index.json 条目数   = {len(wm)}")
print(f"  最小/最大分片参数   = {min(per.values())/1e6:.1f}M / {max(per.values())/1e6:.1f}M")
PY

echo
echo "############ B) 显卡 ############"
echo "  /dev/nvidia* : $(ls /dev/nvidia* 2>/dev/null | tr '\n' ' ' || echo '（无设备节点）')"
echo "  驱动版本     : $(cat /proc/driver/nvidia/version 2>/dev/null | head -1 || echo 未知)"
echo "  nvidia-smi   :"
(nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv,noheader 2>&1 | head -4) | sed 's/^/    /'
echo "  torch 视角   : $($PY -c "import torch;print('cuda avail', torch.cuda.is_available(), '| build', torch.version.cuda)" 2>&1 | tail -1)"

echo
echo "############ C) 机器上还有没有别的模型 ############"
echo "  --- \$R/models/ 全部内容 ---"
ls -la "$R/models/" 2>/dev/null | sed 's/^/    /'
echo "  --- 全盘搜 *.safetensors / *.gguf / *.bin（排除目标模型）---"
find / -xdev \( -name "*.safetensors" -o -name "*.gguf" \) 2>/dev/null \
  | grep -v "/models/Qwen3.8-27B/" | head -20 | sed 's/^/    /'
echo "    （以上为空 = 只有目标模型）"

echo
echo "############ D) 进程 / venv / 缓存 ############"
echo "  --- vllm / python 常驻进程 ---"
pgrep -af "vllm|venv_q38|_q38_" 2>/dev/null | sed 's/^/    /' || echo "    （无）"
echo "  --- venv 目录 ---"
ls -d "$R"/venv* 2>/dev/null | sed 's/^/    /' || echo "    （无）"
echo "  --- 磁盘占用明细 ---"
du -sh "$R"/* 2>/dev/null | sort -rh | sed 's/^/    /'
echo "  --- 磁盘余量 ---"
df -h "$R" | tail -1 | sed 's/^/    /'

echo
echo "############ E) 可清理的临时产物 ############"
for p in "$R/tmp/uv.whl" "$R/tmp/probe" "$R/tmp/probe.log" "$R/tmp/_spd.bin" \
         "$R/tmp/_q38_files.tsv" "$R/_q38_dl.out"; do
  [ -e "$p" ] && echo "    $(du -sh "$p" 2>/dev/null | cut -f1)  $p"
done
echo "    $(du -sh "$R/pipcache" 2>/dev/null | cut -f1)  $R/pipcache（已弃用：改用 uv 后不再需要）"
echo "=== DONE ==="

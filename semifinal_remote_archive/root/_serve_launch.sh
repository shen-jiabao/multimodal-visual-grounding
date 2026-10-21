#!/bin/bash
# 单卡 96 GB 起服：把 `_get_serve_q38.sh` 包成**后台分离**任务，断线也不中断。
# 起服机制本身不重写（本项目纪律：判据只实现一处）。
set -u
R=/root/rivermind-data

echo "=== 前置 ==="
echo "  /dev/nvidia*: $(ls /dev/nvidia[0-9]* 2>/dev/null | tr '\n' ' ' || echo 无)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,compute_cap \
  --format=csv,noheader 2>&1 | sed 's/^/  /'

if pgrep -f "_serve_run.sh" >/dev/null 2>&1; then
  echo "起服流程已在运行（pid $(pgrep -f _serve_run.sh | tr '\n' ' ')）"; exit 0
fi

echo
echo "=== 清旧进程 ==="
pkill -f "_serve_run.sh" 2>/dev/null
pkill -f "vllm serve" 2>/dev/null
sleep 4
pkill -9 -f "vllm serve" 2>/dev/null   # 实测 SIGTERM 有时杀不掉，补一刀
sleep 2
if pgrep -af "vllm serve|_serve_run.sh" >/dev/null 2>&1; then
  echo "  ★仍有残留："; pgrep -af "vllm serve|_serve_run.sh" | cut -c1-120 | sed 's/^/    /'
else
  echo "  ✓ 旧进程已清空"
fi
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>&1 | sed 's/^/  显存已用 /'

cat > "$R/_serve_run.sh" <<'EOS'
#!/bin/bash
set -u
R=/root/rivermind-data
{ bash "$R/_get_serve_q38.sh"; } > "$R/_q38_serve.log" 2>&1
echo "=== SERVE-DONE rc=$? ===" >> "$R/_q38_serve.log"
EOS
chmod +x "$R/_serve_run.sh"
: > "$R/_q38_serve.log"

# ★venv/bin 必须进 PATH：vLLM 后端 JIT 编译要调用 `ninja`，而它只装在 venv 里
#   （实测首轮起服就是因 `FileNotFoundError: 'ninja'` 而 EngineCore 初始化失败）。
#   注意：**只在此处注入**，不写进共享的 _q38_env.sh——那个文件被构建等脚本复用，
#   全局前置 venv/bin 会把 `python3` 抢到 venv 解释器上，破坏建环境流程。
export PATH="$R/venv_q38/bin:$PATH"
# ★合并图（5760×1080 ≈ 6120 视觉 token）+ multi_box（20 条 → 输出更长）
#   ⇒ 8192 上下文不够，抬到 16384（96 GB 卡、权重 55.6 GB，KV cache 余量充足）
export Q38_MAXLEN="${Q38_MAXLEN:-16384}"
export Q38_UTIL="${Q38_UTIL:-0.90}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo
echo "=== 启动（Q38_MAXLEN=$Q38_MAXLEN  Q38_UTIL=$Q38_UTIL）==="
echo "  ninja: $(command -v ninja || echo '★仍不在 PATH')"
cd "$R"
setsid nohup bash "$R/_serve_run.sh" > /dev/null 2>&1 < /dev/null &
sleep 8
echo "  已启动 pid $(pgrep -f _serve_run.sh | tr '\n' ' ')"
echo "  日志 $R/_q38_serve.log ｜ vllm $R/vllm38.log"
echo "  --- _q38_serve.log 首 18 行 ---"
head -18 "$R/_q38_serve.log" 2>/dev/null | sed 's/^/    /'
echo "=== DONE ==="

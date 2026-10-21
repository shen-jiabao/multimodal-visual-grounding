#!/usr/bin/env bash
# =============================================================================
# 现场演示脚本 · 服务器只读取证（答辩用）
# 用法：把本文件放到服务器上，执行  bash demo_server_evidence.sh
#       （或直接把本文件内容整段粘贴到终端）
#
# 安全声明：本脚本**只读**——仅使用 stat / head / tail / grep / sha256sum / ls / du，
#           不写入、不删除、不移动任何文件。
# 预期耗时：约 40 秒（含人工讲解停顿）
# =============================================================================
set +e
R=/root/rivermind-data
cd "$R" 2>/dev/null || { echo "找不到 $R"; exit 1; }

line() { printf '%s\n' "------------------------------------------------------------"; }
step() { echo; line; echo "【$1】$2"; line; }

echo "======================================================================"
echo " 答辩现场取证 · 只读演示      $(date '+%Y-%m-%d %H:%M:%S %z')"
echo " 主机: $(hostname)"
echo "======================================================================"

# ---------------------------------------------------------------------------
step "① 上游产物确实在这台机器上（纳秒级时间戳）"
echo '\$ stat -c "%y | %s B | %n" q38_dl.log _q38_serve.log vllm38.log _q38_gate20.json'
for f in q38_dl.log _q38_serve.log vllm38.log _q38_gate20.json _qwen_full_pipeline.py; do
  [ -e "$f" ] && stat -c '   %y | %10s B | %n' "$f"
done
echo "  ↑ 时间戳精度到纳秒；提交件 0.7963 由这些产物装配而来"

# ---------------------------------------------------------------------------
step "② 权重是逐片「大小 + sha256」双验过的（不是"看起来下完了"）"
echo '$ tail -3 q38_dl.log'
tail -3 q38_dl.log
echo
echo '$ head -2 q38_dl.log'
head -2 q38_dl.log

# ---------------------------------------------------------------------------
step "③ 起服时确认过是自己要的模型（防"起错模型"）"
echo '$ grep -E "qwen38-27b|Qwen3.8-27B|分片" _q38_serve.log'
grep -E 'qwen38-27b|Qwen3.8-27B|分片' _q38_serve.log

# ---------------------------------------------------------------------------
step "④ vLLM 真实跑过：参数 / 架构 / 显存 / 毫秒级时间戳"
echo '$ grep -E "Resolved architecture|Model loading took|Available KV cache memory" vllm38.log'
grep -E 'Resolved architecture|Model loading took|Available KV cache memory' vllm38.log

# ---------------------------------------------------------------------------
step "⑤ 同一份权重：当场比对指纹（应与包内 模型文件/model_sha256.txt 一致）"
echo '$ sha256sum models/Qwen3.8-27B/config.json'
sha256sum models/Qwen3.8-27B/config.json 2>/dev/null
echo "  预期：191e0af2…3aab（本包 model_sha256.txt 中同值）"
echo
echo '$ ls models/Qwen3.8-27B | wc -l   # 分片+配置+分词器 共 31 项'
ls models/Qwen3.8-27B | wc -l

# ---------------------------------------------------------------------------
step "⑥ 数据与作用域词表：与包内同源"
echo '$ sha256sum dataset/queries/queries.json _ordinal_cores_r2.json'
sha256sum dataset/queries/queries.json _ordinal_cores_r2.json 2>/dev/null
echo "  预期：queries 717d15a1…95c4　|　cores cfba587d…80c2（与包内一致）"
echo
echo '$ for m in visible infrared depth; do echo -n "$m: "; ls dataset/Images/$m | wc -l; done'
for m in visible infrared depth; do echo -n "  $m: "; ls dataset/Images/$m 2>/dev/null | wc -l; done

# ---------------------------------------------------------------------------
step "⑦ 现场互动：请评委任点一个文件，当场比对"
cat <<'TIP'
 把下面两行发给评委（让 TA 报一个文件名，例如 q38_dl.log）：
     stat -c '%y | %s B | %n' <文件名>
     sha256sum <文件名>
 然后翻到本包 代码与数据/自检/包内文件清单_SHA256.txt 找同名行比对。
TIP

echo
echo "======================================================================"
echo " 取证结束（全程仅 stat/head/tail/grep/sha256sum/ls，未修改任何文件）"
echo "======================================================================"

#!/bin/bash
# 预登记 M 发车：5690 全量、三模态分图、5 条规则、multi_box、max_new=1024、thinking 关闭。
#
# 不重写启动逻辑：判据/闸门全部复用 `_launch_q38.sh`（本项目纪律：判据只实现一处），
# 本脚本只负责**注入 M 件的环境变量**。
set -u
R=/root/rivermind-data

# 作用范围：M 件跑全量（含序数 1103），故关掉"只跑非序数"
export QWEN_ONLY_NONORD=0
# 透传给管线的 M 开关（`_launch_q38.sh` 会整体拼进 ENVS）
# ★ 2026-10-05 用户指定：三模态**合成一张图**（"3m" = merge3 横向三联），每次只传这一张。
# ★ QWEN_MERGE_SPACE=whole 的由来（几何判据，非分数）：20 条冒烟里模型给的是
#   **整张合并图**的 0–1000（多处"占满可见光格"的目标 x2 恰为 0.333 = 1/3）。
#   拿交付件(0.7963)框当独立参照比 IoU：panel 中位 0.0000（0/19 >0.5），
#   whole(x×3) 中位 0.9032（17/19 >0.5）⇒ 判 whole。见预登记 M §3.5。
export QWEN_EXTRA="QWEN_MODALITIES=3m QWEN_RULES5=1 QWEN_MULTIBOX=1 QWEN_MAXNEW=1024 QWEN_THINKING=0 QWEN_MERGE_SPACE=whole"
# 产物单列，绝不覆盖 I 件 q38 的产物
export QWEN_OUT="$R/output/m_direct.json"
export QWEN_OUT_SOM="$R/output/m_som.json"
export QWEN_LOG="$R/m_run.log"

echo "=========== 预登记 M 发车 ==========="
echo "  作用范围 : 全量 5690（QWEN_ONLY_NONORD=$QWEN_ONLY_NONORD）"
echo "  透传开关 : $QWEN_EXTRA"
echo "  产物     : $QWEN_OUT"
echo "  日志     : $QWEN_LOG"
echo "===================================="

bash "$R/_launch_q38.sh"
rc=$?

# ★发车自检（教训固化）：开关是"经 _launch_q38.sh 拼进 ENVS"传下去的，
#   一旦远端 launcher 是旧版就会**静默丢失**（实测发生过一次：跑成了
#   MODALITIES=1 / 无规则 / 无 multi_box 的错误配置）。故启动后必须回读
#   管线自己的 [pipe] 行，确认它真的以 3m 在跑；不符则立刻停。
echo
echo "=========== 发车自检 ==========="
sleep 5
pipe_line=$(grep -m1 "\[pipe\] 模态=" "$QWEN_LOG" 2>/dev/null)
echo "  管线首行: ${pipe_line:-（未出现）}"
if [ "$rc" -ne 0 ]; then
  echo "  ★_launch_q38.sh 退出码 $rc（闸门未过）⇒ 不发车"
  exit "$rc"
fi
if echo "$pipe_line" | grep -q "模态=3m" \
   && echo "$pipe_line" | grep -q "multi_box=True" \
   && echo "$pipe_line" | grep -q "规则5=True"; then
  echo "  ✓ 自检通过：管线以 3m + 规则5 + multi_box 在跑"
else
  echo "  ★★自检失败：开关未生效 ⇒ 立即停止全量"
  pkill -f "_qwen_full_pipeline.py" 2>/dev/null
  echo "  --- 日志前 8 行 ---"
  head -8 "$QWEN_LOG" 2>/dev/null | sed 's/^/    /'
  exit 9
fi

echo
echo "=========== 运行中快照 ==========="
echo "  进程: $(pgrep -f _qwen_full_pipeline.py | tr '\n' ' ')"
echo "  --- 日志末 6 行 ---"
tail -6 "$QWEN_LOG" 2>/dev/null | cut -c1-150 | sed 's/^/    /'

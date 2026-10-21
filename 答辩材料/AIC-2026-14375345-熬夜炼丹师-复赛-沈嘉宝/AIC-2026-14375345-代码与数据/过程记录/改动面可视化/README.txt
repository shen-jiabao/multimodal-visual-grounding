改动面可视化（交付件 vs 其装配底座）
=====================================

用途：把交付件 `result_semi_q38_nonord`（0.7963）与其装配底座
`result_semi_n32bf16_nonord`（0.7786）**逐条比对**，抽样渲染成"改动面拼版图"，
供人眼复核"被替换的 519 处框是否落在查询描述的目标附近"。

着色规则（脚本 `code/_viz_q38.py`）：
  · 蓝框 = 基线（0.7786 件）的框
  · 红框 = 本交付件（0.7963 件）的框
  · 每张子图下方附 **qid、两框互 IoU、查询原文**

文件：
  _sheet_changed.png   20 处**真分歧**样例（两框互 IoU < 0.5，即两件指向不同位置）
  _sheet_kept.png      20 处**未改动**对照样例（两框 IoU ≥ 0.5）

重渲染命令（需赛事数据集图像；本队工作区实测通过，耗时约 1 分钟）：
  cd AIC-2026-14375345-代码与数据
  set AIC_ROOT=<放有 复赛数据集-基于大模型的多模态视觉理解与推理/ 的目录>   # Windows
  python code/_viz_q38.py --piece result_semi_q38_nonord \
         --base result_semi_n32bf16_nonord \
         --out-dir <输出目录> --mode entries --n 20 --seed 0

本次复核结论（可复算）
----------------------
1) 交付件相对 0.7786 基线的**真分歧条数 = 519**（判据 IoU < 0.5），
   与装配器打印的「★改动 519」**完全一致**。
   复算脚本：
     python - <<'PY'
     import json,zipfile
     n=json.load(zipfile.ZipFile('../AIC-2026-14375345-赛题结果文档/result_semi_q38_nonord.zip').open('result_semi_q38_nonord.json'))
     b=json.load(open('submission_zips/semifinal/0.7786/result_semi_n32bf16_nonord.json'))
     def iou(a,c):
         ix=max(0,min(a[2],c[2])-max(a[0],c[0])); iy=max(0,min(a[3],c[3])-max(a[1],c[1])); i=ix*iy
         u=(a[2]-a[0])*(a[3]-a[1])+(c[2]-c[0])*(c[3]-c[1])-i
         return i/u if u>0 else 0
     print(sum(1 for k in n if iou(n[k]['bbox'],b[k]['bbox'])<0.5))   # ⇒ 519
     PY
2) 本拼版所用的 20 张 changed 样例，**100% 落在上述 519 条分歧集之内**；
   20 张 kept 样例与分歧集**交集为 0**（即抽样标注与实际改动状态一致）。
3) 重渲染（同命令、同 seed）得到的 changed 样例 qid 与本目录拼版**逐一相同**：
   000022_006 / 000540_001 / 000911_001 / 001044_013 / 001352_004 / 001562_008 …
   ⇒ 本拼版确为该命令的产物，可重放。

说明：本目录**不含**逐条单图（40 张，约 16 MB），只保留两张拼版以控制包体；
单图可由上面的重渲染命令原样生成。

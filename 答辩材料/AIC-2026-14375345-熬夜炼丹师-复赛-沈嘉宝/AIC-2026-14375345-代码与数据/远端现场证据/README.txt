远端现场证据（只读采集）
=====================

来源：本队赛程期间使用的云端 GPU 实例（AutoDL 容器 autodl-container-a19d43b48e-bedc3b24，
SSH 端口 43504）。目录 /root/rivermind-data 为本项目在该实例上的工作目录。
采集方式：SFTP 只读下载 + 远端 sha256 清单比对（未修改、未删除远端任何文件）。

本目录的用途：作为「交付件 0.7963 的 Qwen3.8-27B 路线」在**真实运行环境**中的现场佐证——
证明该环境为本队自建、模型为本队自下载并逐片校验、闸门与起服均为本队脚本实际执行过。

文件清单（本地文件  →  远端路径，sha256 与远端逐位一致）
--------------------------------------------------------
远端文件清单_SHA256.txt   远端只读采集的清单（含字节数 / mtime / sha256），本目录所有文件的
                          指纹均可在其中核对。
q38_env.log               /root/rivermind-data/q38_env.log    环境构建实录：
                          · vLLM registry 登记检查：架构总数 365，Qwen3_5ForConditionalGeneration 已登记
                          · torch 2.11.0+cu130（无卡时 cuda avail=False 属正常）
_q38_build.out            /root/rivermind-data/_q38_build.out 独立 venv（venv_q38）构建输出
q38_dl.log                /root/rivermind-data/q38_dl.log     权重下载逐片记录：
                          · 18 个分片「大小 + sha256 双验通过」，最后一行「全部 18 个分片就位」
vllm38.log                /root/rivermind-data/vllm38.log     vLLM 起服完整日志（v0.23.0）：
                          · model = /root/rivermind-data/models/Qwen3.8-27B
                          · served_model_name = qwen38-27b
                          · Resolved architecture: Qwen3_5ForConditionalGeneration
                          · disable_custom_all_reduce = True（cu130 + A100 兼容处置）
                          · limit_mm_per_prompt = {image:1, video:0}
                          · dtype=torch.bfloat16；Model loading took 51.08 GiB
_q38_serve.log            /root/rivermind-data/_q38_serve.log 起服自证：
                          · 就绪后回读 /v1/models ⇒ id = qwen38-27b、root = models/Qwen3.8-27B
                            （防止「起错模型」这一最严重的事故）
                          · 含「下一步」清单：20 条冒烟 → 全量 4587 → 装配（锚 ord 689 / 非序数 3741）
_q38_gate20.json          /root/rivermind-data/_q38_gate20.json      闸门 G2 冒烟产物：20 条查询的
                          解析框 + 模型原始输出（raw），用于机检「可解析率」
_q38_gate20_panel.json    /root/rivermind-data/_q38_gate20_panel.json 闸门 G3 目检数据
q38_gate20_sheet.png      /root/rivermind-data/q38_gate_vis/_sheet.png G3 目检 4×5 拼版图
                          （一屏看完 20 条框是否落在查询描述的目标附近）
远端原件脚本/              /root/rivermind-data/ 下的原始脚本（含 /root 硬编码路径，未做可移植化）：
  _qwen_full_pipeline.py    主管线（Phase A 直接定位 / Phase B 序数 SoM；api32b 引擎）
  _get_serve_q38.sh         交付件级起服脚本（目录断言 / TP 判定 / 权重完整性 / 逐后端重试）
  _launch_q38.sh            全量启动器（五道闸门全过才发车）
  _q38_gate.py              G2/G3 闸门实现
  _q38_gate.sh              闸门冒烟包装
  _q38_build.sh / _q38_env.sh / _q38_dl.sh   环境构建 / 环境检查 / 权重下载
  _make_submission.py       装配提交 json 与 zip

注：随包 code/remote_shell/ 下的同名脚本是上述远端脚本的**原样副本**（仅加 ROOT 注入的为该目录
以外的 Python 脚本）；远端原件与随包副本的 sha256 差异即为「路径可移植化」这一处改动。

权重与数据的同源核验（远端 ⇄ 本包）
------------------------------------
  models/Qwen3.8-27B/config.json          191e0af2…3aab   （= 模型文件/model_sha256.txt）
  models/Qwen3.8-27B/tokenizer.json       0997f410…b9f3
  models/Qwen3.8-27B/model.safetensors.index.json  77042094…b9df
  dataset/queries/queries.json            717d15a1…95c4   （= 代码与数据/复赛数据集-…/queries/queries.json）
  _ordinal_cores_r2.json                  cfba587d…80c2   （= 代码与数据/_ordinal_cores_r2.json）
  dataset/Images/visible                  1295 张（与 queries.json 的被引用图数一致）

说明（如实陈述）
----------------
本实例（43504）自身于 2026-10-05 创建，其现场记录承载了 Qwen3.8-27B 环境构建、闸门与起服的
完整过程；本交付件的装配输入 `_qwen38_direct.json`（sha256 E561606C…CCBF0）已在本地留存并随包
（代码与数据/_qwen38_direct.json），可由本包的装配器离线逐条复算为最终提交件——见
顶层《比赛流程与自证材料.md》§5.4。

补充澄清（避免误读为"自相矛盾"）
--------------------------------
1) **两台实例的 GPU 不同，均属正常**：
   · 本现场实例（43504，2026-10-05）：**单卡 NVIDIA RTX PRO 6000 Blackwell，97887 MiB**，
     故起服自动判定 **TP=1**、maxlen=16384 —— 证据见 `_q38_serve.log` 第 6 行与 `vllm38.log`；
   · 交付件阶段（`env_record.txt` STAGE-2，记录时间 2026-10-03 18:14 UTC）：**NVIDIA A100-SXM4-40GB**
     （`env_record.txt` 第 26 行 `NVIDIA A100-SXM4-40GB, 580.126.09, 40960 MiB, 8.0`），
     其服务命令行为 `--tensor-parallel-size 2`（51.77 GiB 权重单卡 40 GB 放不下 ⇒ 必为双卡）。
   ⇒ 二者是**赛程中先后使用的不同实例**；本目录只作为"环境是自建、模型是自下、闸门是自跑"的现场佐证。

2) **归档内个别文件扩展名与内容不符**（远端原件如此，未作改动以保同源）：
   `运维脚本归档/_probe_f.py` 实为 bash 脚本（首行 `#!/bin/bash`）。
   另：`运维脚本归档/_q38_dl_launch.sh` 的注释写"ModelScope 通道"，而实际下载脚本
   `_q38_dl.sh` 走的是 **hf-mirror.com（HuggingFace 镜像）+ 逐片 sha256 校验** ⇒ **以 `_q38_dl.sh` 实现为准**。


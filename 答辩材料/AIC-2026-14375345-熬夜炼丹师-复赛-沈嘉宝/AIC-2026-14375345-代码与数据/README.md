# 基于大模型的多模态视觉理解与推理 · 代码与数据说明

> 参赛团队编号 **AIC-2026-14375345** ｜ 团队名称 **熬夜炼丹师** ｜ 队长 沈嘉宝
> 交付件：`../AIC-2026-14375345-赛题结果文档/result_semi_q38_nonord.zip`（内部同名 JSON，5690 条）
> 线上 **ACC@0.5 = 0.7963**

---

> ## ★ 本包状态（先读这一条）
>
> **本包的技术路线 = `result_semi_q38_nonord.zip`（0.7963）的技术路线**，即
> **「跨代换模型 Qwen3.8-27B 重跑非序数层」**：非序数 4587 条用 Qwen3.8-27B 直接定位，
> 经六桶互斥装配器替换 **519** 处（净 +101 条），序数层 1103 条**逐位不动**。
>
> 本目录以这条路线为**交付件路线**；其它路线（L7 异质模型 72B 等）仅作历史留档随包
> （`submission_zips/semifinal/待提交/`、`_k*.json`），**不作为交付件**。
> 装配出的交付件可**离线逐条复算**（见 §5 第 3 条，已实测与交付件逐条一致）。

---

## 一、项目简介

### 1.1 任务

给定同场景的**可见光（visible）/ 红外（infrared）/ 深度（depth）**三模态图像与一条英文查询，
输出目标在**可见光图**中的归一化边界框 `[x1, y1, x2, y2]`（取值 0–1）。
评测指标 **ACC@0.5**：预测框与真值 IoU ≥ 0.5 记为该条命中。

复赛规模：**5690 条查询 / 1295 张图**（三模态各被引用 1295 张，缺失 0、损坏 0）。
其中**序数类查询 1103 条**（"从右数第二个"这类，占 19.4%），非序数 4587 条。

### 1.2 方案一句话

**不做端到端训练**：以开源多模态大模型承担**感知（直接定位）**，以**确定性规则**承担
**推理（序数指代）**，以**带断言的装配器**承担**出件**。

非序数层（4587 条，唯一作用层）逐级做**单变量替换**——提示词、解码（`temperature=0` 贪心）、
并发（`CONC=8`）与上一级**逐字相同**，只换模型或精度；序数层（1103 条）另用
「查询文本程序化推算 + Set-of-Mark 选号」四层叠加规则；最后经六桶互斥装配器与硬约束校验出件。

### 1.3 成绩沿革（线上 ACC@0.5）

| 版本 | 得分 | 机制 | 改动面 → 净翻转 |
| --- | --- | --- | --- |
| `result_final_semi` | 0.7459 | 8B 全量 + 补框链 | — |
| `result_semi_ord_det` | 0.7552 | ＋序数层水平轴文本推算 | 136 → +53 |
| `result_semi_ord_det_ss` | 0.7568 | ＋候选自排序 | 41 → +9 |
| `result_semi_ord_det_ss_qfb` | 0.7573 | ＋回退点改用直接框 | 47 → +3 |
| `result_semi_ord_det_ss_y_qfb` | 0.7578 | ＋垂直序数轴 | 10 → +3 |
| `result_semi_enum_trunc` | 0.7598 | ＋枚举截断组修复 | 52 → +11 |
| `result_semi_n32_ordbase` | 0.7742 | 32B-AWQ 重跑 4587 条**非序数** | 467 → +82 |
| `result_semi_n32_orddirect` | 0.7766 | 序数层同机制替换（B′） | 87 → +14 |
| `result_semi_n32bf16_nonord` | 0.7786 | 同族**精度修复**（32B bf16 重跑非序数） | 111 → +11 |
| **`result_semi_q38_nonord`** | **0.7963** | **跨代换模型（Qwen3.8-27B 重跑非序数）** | **519 → +101** |

**交付件读数（归零法实测，恒等式）**：

```text
提交分 = (非序数正确条数 + 序数正确条数) / 5690        1 条 = 0.000176
0.7963 × 5690 = 4531 = 非序数 3842 + 序数 689       Δ = +101 条 vs 0.7786 基线
同机制边际成功率（四次独立测量）：17.6% / 16.1% / 9.9% / 19.4%（改动面 467 / 87 / 111 / 519）
```

---

## 二、模型权重下载链接

本项目**只使用开源模型**，未调用任何闭源在线 API。

| 模型 | 下载来源 | 用途 | 是否用于交付件 |
| --- | --- | --- | --- |
| **Qwen3.8-27B** | HuggingFace `Qwen/Qwen3.8-27B`（51.77 GB，18 分片） | **跨代换模型**重跑非序数 ⇒ 产出交付件 0.7963（vLLM 0.23.0，TP=2） | ✅ **是（本交付件）** |
| **Qwen3-VL-32B-Instruct（bf16）** | HuggingFace `Qwen/Qwen3-VL-32B-Instruct`（62.1 GB，14 分片） | 交付件的**装配底座**（0.7786 基线）：同族精度修复重跑非序数 | ✅ 是（基线） |
| Qwen3-VL-32B-Instruct-AWQ | ModelScope `QuantTrio/Qwen3-VL-32B-Instruct-AWQ`（19.12 GB，7 分片） | 非序数层首次重跑（0.7742，vLLM 0.11.0） | 历史层 |
| Qwen3-VL-8B-Instruct | ModelScope `Qwen/Qwen3-VL-8B-Instruct` | 主体早期层：直接定位 + 序数 SoM | 历史层 |
| LocateAnything-3B | ModelScope `nv-community/LocateAnything-3B` | 无框条目红外补轮（45 条 → 出框 26） | 历史层 |
| GroundingDINO（SwinT-OGC）+ bert-base-uncased | HF `ShilongLiu/GroundingDINO` / GitHub `groundingdino_swint_ogc.pth` | 三模态早期融合备选分支 | ❌ 未采用（实测 0.4378） |

复现建议：ModelScope 用 `snapshot_download('<仓库 id>', local_dir=...)`；HuggingFace 侧本项目的下载实现见
`code/remote_shell/_q38_dl.sh`（`curl -C -` 断点续传 + 跨分片并发 + 边完成边合并 + 逐片 sha256 双验）。

**模型文件不入包**（19 GB 级大文件）。`model_sha256.txt`（在同级 `../AIC-2026-14375345-模型文件/`）
记录了**实测逐文件 SHA256**（Qwen3.8-27B 18 分片全列），可用于校验下载完整性。

---

## 三、运行配置

### 3.1 三段式运行

| 段 | 内容 | 硬件 |
| --- | --- | --- |
| **A 离线段** | 数据预处理、序数核心词抽取、序数四层规则、枚举合并、装配、读数、硬约束校验 | **纯 CPU**，无需 GPU |
| **B 推理段**（8B / 32B-AWQ 阶段） | Qwen3-VL 直接定位（Phase A）+ 序数 SoM（Phase B） | GPU ≥ 40 GB |
| **B2 推理段**（**交付件 0.7963**） | Qwen3.8-27B 直接定位（Phase A，仅非序数）⇒ 装配替换 | **2×40 GB**（权重 52 GiB，TP=2） |

依赖见 `requirements.txt`（分 **A 离线 / B 推理 / B2 交付件阶段** 三节，版本号为实测值）。
注意 **B 与 B2 不通用**：交付件用 Qwen3.8-27B，需 **vLLM ≥ 0.22.1**（`Qwen3_5ForConditionalGeneration`
自该版本起登记在 registry；另 **transformers 必须 < 5.0**）。

### 3.2 根路径 `AIC_ROOT`

包内所有 Python 脚本的路径均经 `AIC_ROOT` 解析（缺省 = `code/` 的上一级 = **本目录**）。
**在本目录下直接运行即可，无需改任何代码**；数据放在别处时设该环境变量：

```bash
export AIC_ROOT=/path/to/AIC-2026-14375345-代码与数据   # Windows: set AIC_ROOT=D:\...\AIC-2026-14375345-代码与数据
```

### 3.3 目录布局（脚本按此相对位置找文件）

```text
AIC-2026-14375345-代码与数据/                        ← AIC_ROOT
├── code/                                  复赛正式链路（32 Python 脚本）
│   ├── remote_shell/                      远端起服与闸门 shell（11 个，原样复制、不做路径注入）
│   └── training_未采用/                    训练与评估脚手架（8 个，未采用）
├── 复赛数据集-基于大模型的多模态视觉理解与推理/queries/queries.json  查询索引（1.2 MB，键集来源）
├── submission_zips/semifinal/             各版本提交件（按线上得分归子目录）＋ SHA256.txt
│   ├── 0.7459/ … 0.7963/result_semi_q38_nonord.json
│   └── 待提交/result_semi_k72b_n519.json   未提交变体（留档，非交付件）
├── 判据预登记/                            判据预登记件（17 份：判据写于结果之前）+ 跑后实录
├── 远端现场证据/                          现场证据：起服日志 / 闸门 G2·G3 产物 / 目检图 / 远端原件脚本 /
│                                          运维脚本归档（246 个，本队实例 /root 镜像）+ 只读清单
├── 过程记录/                              消融实验复盘 / 赛事数据核验 / 工作区过程痕迹 / 改动面可视化拼版
├── 自检/                                  交付包自检脚本 _verify_package.py + 包内文件清单_SHA256.txt
├── 初赛提交谱系_SHA256.txt                 初赛阶段（9555 条）12 次提交的得分与 SHA256
├── _*.json                                算法自产中间产物（供离线复算，见 §5 第 3 条）
├── env_record.txt                         环境实录（STAGE-1 + STAGE-2）
├── README.md / requirements.txt / 复现流程.md / 技术报告素材_方法学结论.md
├── 技术方案_0.7963路线.md                  技术方案文档源（与顶层 PDF 同源）
└── 代码与数据说明.txt                       目录与产物来源说明
```

> 说明：为保持**代码零改动即可运行**，随包产物按脚本预期布局放置——
> 中间产物 `_*.json` 与查询索引均置于 `AIC_ROOT` 之下，历史版本件置于
> `submission_zips/semifinal/<得分>/`（由 `code/_sub_path.py` 统一解析）。

### 3.4 实测环境（完整记录见 `env_record.txt`）

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA A100-SXM4-40GB（compute capability 8.0，40960 MiB）；交付件阶段服务命令行为 `--tensor-parallel-size 2` ⇒ **双卡**（51.77 GiB 权重单卡放不下）。卡间互联方式未记录（`env_record.txt` 无 topo 输出）。另：现场佐证实例为**单卡 RTX PRO 6000 Blackwell 96GB**（TP=1），见 `远端现场证据/README.txt` |
| 驱动 / CUDA | 580.126.09 / driver CUDA 13.0 |
| 系统 / Python | Ubuntu 22.04.5 LTS / Python 3.10.12 |
| 推理栈（8B / 32B-AWQ 阶段） | vLLM **0.11.0**、transformers **4.57.6**、torch **2.8.0**（cu128）、triton 3.4.0 |
| 推理栈（交付件 0.7963 阶段） | 独立 venv `venv_q38`：vLLM **0.23.0**、transformers **4.57.6**、torch **2.11.0+cu130** |
| 本地离线段 | Windows + Python 3.12；numpy 2.5.1、pillow 12.2.0、paramiko 5.0.0 |

**交付件阶段起服要点（实测踩坑，已固化在 `code/remote_shell/_get_serve_q38.sh`）**：

1. `--disable-custom-all-reduce`：`cu130 + A100(cc8.0) + 驱动 580` 组合下 custom all-reduce 内核在
   CUDA graph 捕获阶段崩（`custom_all_reduce.cuh:455 'invalid argument'` → TP1 死）。退回 NCCL 即可，
   **不影响提示词/解码/并发**（单变量前提不受影响）。
2. `--limit-mm-per-prompt '{"image":1,"video":0}'`：否则编码器按"最大尺寸视频"探测，一次性分配 5.77 GiB 而 OOM。
3. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`：缓解碎片。
4. flashinfer JIT 需要 nvcc 与 dev 符号链接：pip 版 CUDA 只装 `libX.so.N` 而不装 `libX.so`
   ⇒ `-lcudart` 找不到；需批量补 `libX.so → libX.so.N` 并把 pip CUDA 链到 `/usr/local/cuda`。

---

## 四、项目文件介绍

### 4.1 `code/` — 复赛正式链路（32 个 Python 脚本，按职责分组）

**共享引擎与路径库**

| 文件 | 作用 |
| --- | --- |
| `_fuse_qwen.py` | **融合判据的唯一实现**：提示词模板、框解析、`usable/iou/norm/area/ordered` 几何原语、`expected_ordinal_num`（四前提文本推算）、`det_ordinal`、`ordered_candidates`、`untrusted_ordinals`。复赛各脚本均导入本模块，避免判据两处实现而漂移 |
| `_sub_path.py` | 提交件路径解析（`find` 读 / `out_path` 写）；"同一提交件不得出现两份"这条约定由它保证 |
| `_fuse_semi.py` | 8B 主模型与 LocateAnything 的互印证取框，产出首版 `result_final_semi` |

**数据预处理**

| 文件 | 作用 |
| --- | --- |
| `_analyze_ordinal3.py` | 序数核心词抽取（复赛 1103 条）。**实测可逐条重建随包词表** |
| `_survey_modality_semi.py` | 复赛查询**类型分类**（8 类，优先级唯一归类、互斥自检）+ 三模态需求分析 |
| `_survey_modality.py` | 初赛口径的同类统计（3 个二值标记） |

**序数层确定性规则**

| 文件 | 作用 |
| --- | --- |
| `_variant_det_semi.py` | 序数四层叠加：`--which det` 水平轴文本推算、`--selfsort` 候选自排序、`--yaxis` 垂直轴、`--qfb` 回退点改用直接框。各层改动集互不相交 |
| `_variant_depth_semi.py` | 深度轴序数变体（读 16 位深度图排各候选深度）；**实测负，未采用**，留档 |

**枚举重跑与合并**

| 文件 | 作用 |
| --- | --- |
| `_enum_retry.py` | 远端：对枚举召回不足的条目换提示词重跑枚举，采集原始框 |
| `_enum_merge.py` | 本地：合并枚举重跑结果、保真度核对、截断层出件（产出 0.7598 件） |

**GroundingDINO 备选分支（留档，未采用）**

| 文件 | 作用 |
| --- | --- |
| `predict_gd_3modal.py` | 三模态早期融合推理（需 torch + GroundingDINO 源码与权重） |
| `_gd_enum.py` | 采集 GD 候选框池 |
| `_gd_merge.py` | GD 候选池的阈值 + NMS 取舍与出件（实测 0.4378） |

**主模型推理（需 GPU）**

| 文件 | 作用 |
| --- | --- |
| `_qwen_full_pipeline.py` | **主管线**：Phase A 全量直接定位 + Phase B 序数 SoM（枚举 → 叠编号 → 选号）。双 checkpoint 断点续跑、闸门与并发均在内部；引擎可切 `hf8b`（本地 transformers）/ `api32b`（vLLM OpenAI 兼容接口）。**交付件由本脚本以 `api32b` + Qwen3.8-27B 产出** |
| `_probe_models.py` | 双模型探针（32B vs 8B），同一批条目、同一套提示词，为模型选型提供实测依据 |
| `_smoke_qwen32b.py` | 接线自校验：我方 `gen` 与归档 `Api32B` 在同一服务器上逐位比对，隔离"接线错误"与"量化漂移" |

**装配 / 读数 / 校验**

| 文件 | 作用 |
| --- | --- |
| `_assembly_32b.py` | **★ 交付件装配器**（产出 0.7963 件）：互斥六桶 + 总和断言；无框 / 退化 / 面积超限 / 过小一律**保留基线**（覆盖率优先） |
| `_assembly_ord_som.py` | 序数层 SoM 装配（产出 0.7729 件，实测净 −7，止损已触发） |
| `_assembly_ord_prime.py` | 序数层「32B 直接框替换同机制基线」装配（产出 0.7766 件） |
| `_ord_layer_levers.py` | **离线杠杆量化**：同一把尺子对比各条路线的改动面与天花板 |
| `_test_assembly.py` | 装配器 8 组受控夹具、20 项断言（把"承诺"变成可判定断言） |
| `_measure_zero.py` | **归零法测量**：生成"测量件"以读出某子集的真实正确条数（§1.3 全部数字的来源） |
| `_d_measure.py` | 三模态小样/全量的**判据唯一实现**：改动率与改动条目 IoU 分布 |
| `_validate_semi.py` | 硬约束校验器（条数 / 键集 / 四字段回显 / bbox 合法性），退出码 0/1 |
| `_q38_gate.py` | **跨代换模型路线的闸门 G2/G3**：20 条冒烟，机检可解析率（G2）+ 生成 4×5 拼版目检图（G3 人眼判据） |
| `_measure_zero_sup646_q38.py` | **缺口定位测量件**（基线换成 0.7963 件）：零化 646「空间最高级」层，读出该层当前正确数 |
| `_q38_next_step.py` | **零 GPU 离线分析**：改动面分层、序数层同机制作用域测算、各方案量化预估 |

**异质模型路线（留档，L7 未提交）**

| 文件 | 作用 |
| --- | --- |
| `_k_scope429.py` | 作用域切分：复现装配器六桶分类，导出 L（429）/ N（519）两个作用域清单；逐桶自检 |
| `_k_scale_diag.py` | 坐标空间判据：从闸门产物 raw 文本重解析整数，比较 H1(÷1000)/H2(÷原图)/H3(÷smart-resize) 三假设 |
| `_viz_q38.py` | 改动面可视化：蓝=基线、红=本件，图下附查询原文 |

**远端通道（工程与运维）**

| 文件 | 作用 |
| --- | --- |
| `_rc.py` | 远端执行 / 上传 / 下载通道。**凭据只从环境变量读取，绝不落盘** |
| `_up_semi.py` | 复赛数据集并行断点续传上传（多 SFTP 连接 + 工作队列） |

### 4.1b `code/remote_shell/` — 远端起服与闸门（11 个，**原样复制，不做路径注入**）

| 文件 | 作用 |
| --- | --- |
| `_get_serve_q38.sh` | **★ 交付件级起服脚本**：目录名断言 → GPU 前置检查与 TP 自动判定 → 权重完整性 → flashinfer/nvcc 修复 → `--disable-custom-all-reduce` → 逐后端重试 |
| `_launch_q38.sh` | 全量启动器：**五道闸门**（服务自证 / 管线词表 / 数据齐备 / 输出磁盘 / 管线常量 + G2 冒烟证据），全过才发车 |
| `_q38_gate.sh` | 闸门冒烟包装（调用 `_q38_gate.py`） |
| `_q38_poll.sh` | 起服就绪轮询（以 HTTP 为准） |
| `_q38_dl.sh` | HuggingFace 侧权重下载：断点续传 + 跨分片并发 + 边完成边合并 + 逐片 sha256 双验 |
| `_k_serve.sh` / `_k_poll.sh` | 异质模型（72B）起服与就绪轮询（与 `_get_serve_q38.sh` 同一套机制，只换 `Q38_*` 变量） |
| `_k_gate.sh` | 闸门冒烟（复用与交付件同一批 20 条，保证可比） |
| `_k_fill_launch.sh` / `_k_n_launch.sh` | L 件（429 位）与 N 件（519 位）的启动器 |
| `_k_fill_poll.sh` | 批量进度查询（参数化） |

> 说明：这些脚本内的路径本来就是**远端绝对路径**（`/root/rivermind-data`），
> 跨机可解析性由 `Q38_*` 环境变量提供，**不需要**像 `code/*.py` 那样做 `ROOT` 注入。

### 4.2 `code/training_未采用/` — 训练与评估脚手架（8 个，**未采用**）

`prepare_data.py`、`train.py`、`train_competition.py`、`train_lora_gd.py`、
`train_pose_classifier.py`、`predict.py`、`config.py`、`evaluate.py`。

面向**初赛**口径（9555 条、含真值），仅用于使"预处理→训练→推理→评估"四个环节
在代码上都有对应实现。**本方案最终未采用任何训练权重**，原因见该目录内 README。

### 4.3 结果与档案

| 路径 | 内容 |
| --- | --- |
| **`../AIC-2026-14375345-赛题结果文档/result_semi_q38_nonord.zip`** | **排行榜结果文件**（第 4 项交付物），内含同名 JSON，5690 条 |
| `submission_zips/semifinal/<得分>/` | 各版本提交件（按**线上实际得分**归目录，0.7459 → 0.7963） |
| `submission_zips/semifinal/SHA256.txt` | 提交件谱系：各版本 SHA256、线上读数链与逐条明细 |
| `判据预登记/` | **判据预登记件（17 份）**：作用域 / 闸门 / 事前预期 / 止损线均写在跑之前，可核验产出自主性（指纹见顶层 `比赛流程与自证材料.md` §5.2） |
| `远端现场证据/` | **现场证据（17 份 + 运维脚本归档 246 个 + 服务器取证）**：环境构建实录、权重下载双验、起服自证、vLLM 完整日志、闸门 G2/G3 产物与目检图、远端原件脚本、本队实例 `/root` 运维脚本归档；**`服务器取证/`** 为 2026-10-15 的只读采集（原始输出 + 6 张证据附图：文件时间戳 / 日志原文 / 目录统计 / 开发会话时间线）；逐文件 SHA256 见各目录 `README.txt` 与 `远端文件清单_SHA256.txt` |
| `过程记录/` | 过程文档：`消融实验复盘.md`（初赛 11 版策略消融）、`赛事数据核验.txt`（1295 引用 / 0 缺失 / 1198 png + 97 jpg）、`工作区过程痕迹.md`（开发足迹清单）、`改动面可视化/`（交付件 vs 基线的 519 处改动拼版 + 复核结论） |
| `自检/` | **交付包一键自检**：`_verify_package.py`（指纹 14 项 + 硬约束 4 项 + 改动面读数）+ `包内文件清单_SHA256.txt`（全包指纹台账） |
| `初赛提交谱系_SHA256.txt` | 初赛阶段（9555 条评测集）12 次提交的得分与 SHA256 指纹 |
| `env_record.txt` | 环境实录：驱动 / CUDA / 全量 pip freeze / **模型逐分片 SHA256** / 服务命令行 |
| `技术报告素材_方法学结论.md` | **半决赛第 5 项（PDF 技术报告）素材**：三条方法学结论、数据格式发现、不足之处如实陈述、可复算性索引 |
| `复现流程.md` | **宏观复现手册**：两条复现路径、全链 6 级逐级命令、判据与参数总表、读数口径、验收清单、已知坑速查 |
| `技术方案_0.7963路线.md` | 技术方案文档源（与顶层 `../AIC-2026-14375345-技术方案.PDF` 同源） |
| 本目录 `_*.json` | 算法自产的中间产物（序数核心词表、8B 两阶段输出、枚举重跑、q38 直出、作用域清单等） |

---

## 五、测试命令

以下命令均在本目录（`AIC-2026-14375345-代码与数据/`）下执行。0–4 为**纯 CPU、已实测通过**。

```bash
# 0) ★ 交付包一键自检（指纹 14 项 + 硬约束 4 项 + 改动面读数）
python 自检/_verify_package.py
#    期望末行：结论：✓ 全部通过（指纹 14/14、硬约束 4/4、改动面 519），退出码 0

# 1) 交付件硬约束校验（★ 最快自检）
python code/_validate_semi.py ../AIC-2026-14375345-赛题结果文档/result_semi_q38_nonord.zip
#    期望：条数 5690 / 键集与 queries 一致 / 四字段原样回显 / bbox 全合法，退出码 0

# 2) 装配器夹具回归（8 组夹具、20 项断言，含逐桶计数）
python code/_test_assembly.py
#    期望：全部通过，退出码 0

# 3) ★ 用随包产物**离线复算交付件**（不需要 GPU、不需要重跑推理）
python code/_assembly_32b.py --direct _qwen38_direct.json \
    --base result_semi_n32bf16_nonord --base-score 0.7786 \
    --ord-correct 689 --base-correct 3741 --stem result_semi_q38_nonord --zip
#    期望：六桶 329 / – / 100 / – / 3639 / 519（合计 4587），
#    产物 JSON 与 赛题结果文档/result_semi_q38_nonord.zip 内 JSON **逐条完全一致**（已实测）

# 4) 离线杠杆量化（各路线改动面与天花板，复算 §1.3）
python code/_ord_layer_levers.py

# 5) 数据预处理：重建序数核心词表（应与随包 _ordinal_cores_r2.json 逐条相同）
python code/_analyze_ordinal3.py \
    --dataset "复赛数据集-基于大模型的多模态视觉理解与推理" --out _rebuild_cores.json

# 6) 闸门冒烟（起服后、全量前必跑；G2 机检 + G3 目检图）
python code/_q38_gate.py --n 20 --out /tmp/_q38_gate20.json --vis-dir /tmp/q38_gate_vis
```

**全量推理（需 2×40 GB GPU + 数据集图像；交付件 0.7963 的实际路线）**：

```bash
# ① 起服务（vLLM 0.23.0 + Qwen3.8-27B + TP=2）：远端脚本 code/remote_shell/_get_serve_q38.sh
#    （含 --disable-custom-all-reduce 等实测处置，见 §3.4）
# ② 跑主管线（提示词/解码/并发与上一版逐字相同 ⇒ 单变量：只换模型）
QWEN_ENGINE=api32b \
QWEN_API_BASE=http://127.0.0.1:8000/v1 \
QWEN_API_MODEL=qwen38-27b \
QWEN_CONC=8 \
QWEN_PHASE=A \
QWEN_ONLY_NONORD=1 \
QWEN_DATA="$AIC_ROOT/复赛数据集-基于大模型的多模态视觉理解与推理" \
QWEN_CORES="$AIC_ROOT/_ordinal_cores_r2.json" \
QWEN_OUT_DIRECT="$AIC_ROOT/qwen38_direct.json" \
python -u code/_qwen_full_pipeline.py
#    PHASE=A 只跑直接定位；PHASE=B 跑序数 SoM；ONLY_NONORD=1 只跑非序数（省算力）
# ③ 装配出件（读数三锚点必须与基线件"阶段"对齐 —— 见 §6.4 第 9 条）
python code/_assembly_32b.py --direct qwen38_direct.json \
    --base result_semi_n32bf16_nonord --base-score 0.7786 \
    --ord-correct 689 --base-correct 3741 --stem result_semi_q38_nonord --zip
```

---

## 六、注意事项

### 6.1 关于数据集与权重

- **数据集图像与模型权重均不入包**（大文件）。仅 `queries/queries.json`（1.2 MB）随包，
  供校验器离线自校验；其内容本就逐条内嵌在结果文件里（四字段回显），不构成额外披露。
- 复赛**未提供 `test.json`**（无真值），键集只能取自 `queries.json`。本方案以初赛提交件实测确认
  "提交件键集 == queries 键集 == test 键集"三者相等后才采用。

### 6.2 关于包内代码的一处改动（如实声明）

工作区中每个脚本都硬编码了本机绝对路径 `d:\Desktop\2026AiC`。打包时**仅做路径可移植化**：

- 把该字面量替换为 `ROOT`（`AIC_ROOT` 优先，缺省为 `code/` 的上一级）；
- 在每个脚本头部注入 8 行路径解析（`_HERE` / `ROOT` / `ROOT_S`）；
- `R / "某脚本.py"`（`subprocess` 调用同目录脚本）改为 `_HERE / "某脚本.py"`。

**算法逻辑一行未改**，可用 `diff` 逐文件核对（改动全部落在头部注入块与路径字面量处）。

### 6.3 关于可复现性的边界（重要）

- **离线段 ✅ 可逐字节重建**：预处理、规则层、装配、校验均确定性，已实测
  （本目录 §5 第 3 条：装配产物与交付件逐条完全一致）。
- **推理段 ❌ 不保证逐字节重建**：GPU 推理用 `bfloat16` + 算子调度，浮点归约顺序随硬件、
  驱动、库版本变化；即使代码与权重完全相同，产出也可能有少量差异并向下游传播。
  故本包记录**实测环境**（`env_record.txt`）而非断言可无限复现。

### 6.4 已知坑（复现者可直接避让）

1. **transformers 必须 < 5.0**。vLLM 0.11.0 配 transformers 5.x 会在启动时抛
   `Qwen2Tokenizer has no attribute all_special_tokens_extended`。实测 `4.57.6` 可用。
2. **GroundingDINO 的 `--half`（FP16）不可用**：MSDeformAttn 的 PyTorch 回退不支持 fp16，
   会全量报 `invalid dtype for bias`——表面耗时骤降，实为快速失败。
3. **面积守卫 `MAX_AREA = 0.35`**：超过即视为"合并多个目标"的宽框而拒绝采用。该守卫被全量数据
   两次检验（潜在损失仅约 16–20 条，其中约两成本就是同一个框），故**不放宽**。
4. **新版 PIL 不容忍逆序矩形**（`x1 > x2` 会抛 `ValueError`）。主管线 `draw_som` 已对
   **绘制用**矩形做角点排序；候选表与编号映射不经过该处，**机制未变**。
5. **覆盖率优先**：模型对约 7~9% 的条目输出空数组（认为图中无该物体），
   这类条目**必须保留基线框**——留空必然错，基线框至少还有猜中的机会。
6. **不要随意改写提示词常量**：本项目三次"引入新外部证据"的尝试全部为负
   （−24.8% / −18.6% / −23.1%），成功的杠杆只需"正确候选表 + 框几何"。
7. **"同机制"是可判定条件，不是感觉**：本项目实测教训是「**可迁移的不是规则文本，
   而是被替换那一层的性质**」。替换前请先算一次 `IoU(基线, 拟替换来源)`。
   同机制替换的可用常数（**四次独立测量**）：**≈ +10% ~ +19% × 改动面**
   （467→+82=17.6%；87→+14=16.1%；111→+11=9.9%；**519→+101=19.4%**）；跨机制则 5/5 全负。
   注意 **"模型是否更强"不是必要条件**——19.4% 那次的模型反而更小、官方无接地指标。
8. **数据集有"两个格式子群"，赛题文档只描述了其中一个**（★这条不踩会整批丢分）：
   被引用 1295 张中，**1198 张**属 1920×1080 的 `.png` 组（深度 `I;16` 单通道 uint16、毫米、0–19999），
   **97 张（7.5%）**属 640×360 的 `.jpg` 组（深度为 `RGB` 三通道且**逐像素相同**，实为 **8 位灰度**）。
   直接把 3 维数组交给 `Image.fromarray(arr, mode="L")` 会抛
   `ValueError: Too many dimensions: 3 > 2`，**使那 97 张整批失败**。
   `code/_qwen_full_pipeline.py` 的 `load_depth()` 已按两组合并处理；对 `.jpg` 组**直通原值**。
9. **派生读数必须把锚点与基线件的"阶段"对齐**：归零法的序数锚点（689）是在 **0.7766 阶段**测出的；
   若基线已换成更高分件却沿用旧锚，读数公式就会自相矛盾。每次换基线都要重算锚点并写成硬断言。
   同理，`_assembly_32b.py` 的三个读数参数（`--base-score / --ord-correct / --base-correct`）
   必须与本次基线件成套使用（本交付件用 `0.7786 / 689 / 3741`）。

### 6.5 提交件与命名

- 结果文件为 ZIP，内部含**一个** JSON（名同 zip 基名），5690 条，键集与 `queries.json` 一致，
  `bbox` 为 4 个 0~1 浮点数（`x1<x2, y1<y2`）。
- 提交件按**线上实际得分**归入 `submission_zips/semifinal/<得分>/`，扁平目录只留 `SHA256.txt`。
- 本交付件指纹：`result_semi_q38_nonord.zip`
  SHA256 `FB6C2E471A4F5E8907318FF540A5CAABAEC17A29ECAFC17EB651F512A4AE8F35`（线上 **0.7963**）。

### 6.6 合规

- 仅使用开源模型与开源框架，未调用闭源在线 API；结果全部由算法自主推理产出，后处理为确定性规则。
- 远端访问凭据一律只从环境变量读取、不写入任何文件（见 `code/_rc.py` 模块说明）。

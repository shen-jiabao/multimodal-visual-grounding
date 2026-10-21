# Multimodal Visual Grounding（基于大模型的多模态视觉理解与推理）

> 全球校园人工智能算法精英大赛（AIC 2026）· 赛道一：基于大模型的多模态视觉理解与推理
> 队伍：熬夜炼丹师（AIC-2026-14375345）· 队长：沈嘉宝
> 复赛线上最高成绩：**ACC@0.5 = 0.7963**（交付件 `result_semi_q38_nonord.zip`）

## 1. 任务定义

给定同一场景的**可见光（visible）、红外（infrared）、深度（depth）**三模态图像与一条英文查询，输出目标在可见光图像中的归一化边界框 `[x1, y1, x2, y2]`（坐标除以图像宽/高，取值 0~1）。评测指标为 **ACC@0.5**：预测框与真值框 IoU ≥ 0.5 记为正确，得分为正确条数占比。

三模态数据特性：

| 模态 | 形态 | 判别职责 |
|---|---|---|
| 可见光 | 三通道 8 位 RGB | 主定位依据：类别、颜色、部件等细粒度判据；弱光下细节损失严重 |
| 红外 | 单通道热辐射灰度（三通道堆叠） | 弱光补充：突显活体与热源轮廓，像素值越高温度越高 |
| 深度 | 单通道 16 位，单位毫米（0~19999） | 距离与前后关系：`closest/farthest` 类查询按 ROI 平均深度排序取框 |

数据规模：初赛评测集 9555 条查询；复赛 **1295 张场景图 / 5690 条查询**，其中约 **19.4%** 的查询含序数指代（如 "the third one from the right"）。

## 2. 成绩沿革（线上 ACC@0.5）

- **初赛**：12 次提交，0.6693 → **0.7827**；
- **复赛**：20+ 次提交，0.4378 → 0.7459 → **0.7598**（序数规则层）→ **0.7786**（模型升级/精度修复）→ **0.7963**（跨代换模型 Qwen3.8-27B 重跑非序数层：519 处改动 → 净 +101 条，改动净成功率 19.4%，8.8σ）。

全部提交件按实际得分归档于 `submission_zips/semifinal/<得分>/`，哈希台账见其 `SHA256.txt`。

## 3. 方案路线

采用**预训练多模态大语言模型零样本推理 + 确定性规则层**路线：不做端到端训练、不伪标注、不使用任何真值信息；模型负责语义识别与定位，凡能用纯几何规则确定性表达的部分（序数、方位、相对参照）都交给规则层。

端到端流程（详见 `技术文档.md` §1.3）：

```
[1] queries.json（Query ID → 三模态路径 + 查询文本）
[2] 三模态预处理（归一化 / 红外中值去噪 / 深度离群剔除+掩码 / 弱光融合）
[3] Qwen3.8-27B 前向（multi_box 模式，同图视觉特征 LRU 缓存）
[4] 文本与检测框匹配（三级匹配链 + 空间/序数约束排序）
[5] bbox 归一化与合法性校验（round 4 位，非法回退占位框）
[6] 多产物共识投票后处理（真分歧位 → 满票采纳）
[7] 打包输出 result.zip（内部 submission.json）
```

### 3.1 三模态预处理

- **可见光**：uint8 → float32/255 归一化，统一量纲；
- **红外**：转灰度 → 中值滤波去椒盐噪声 → 复制为三通道伪 RGB；
- **深度**：量程外（0~19999 mm）像素判无效置零，线性归一化并保留有效区掩码，附四级置信度（`high/medium/low/skip`；实测深度无效区占比中位 25.18%，掩码为必备）；
- **弱光增强**：可见光平均亮度低于阈值（60）时，提供红外加权融合（`fuse_rgb_infrared`）与逐像素 Sigmoid 门控融合（`gated_fusion`）两条路径；
- **输入组织**：三模态以分图（`split`）方式按「可见光 → 红外 → 深度」固定顺序送入视觉编码器，提示词附多图角色说明，bbox 一律定义在可见光坐标系。

### 3.2 大模型推理

- 模型：**Qwen3.8-27B-Instruct**（ModelScope `Qwen/Qwen3.8-27B`，27.78B 参数，bfloat16，Safetensors 18 分片约 54GB），`transformers` 原生接口加载，未做任何微调；
- **multi_box 生成**：同图多条查询一次前向生成多框（`max_new_tokens: 1024`），显著减少前向次数；
- **视觉特征缓存**：以图像内容哈希为键的有界 LRU（`cache_max_entries: 4`），同图视觉塔只编码一次——这是全量推理可行时长的决定性加速手段；
- **提示词规则层**：序数扫描（从指定方向边缘起算、只框第 n 个）、参照系判据（物体自身左右 vs 画面左右）、部件只框局部、紧贴目标、颜色为硬约束。

### 3.3 候选匹配与序数层

- **三级匹配链**（`matcher.py`）：精确标签 → 双向单词子集 → 顺序子串；命中多候选且含空间/序数约束时进入确定性选择；
- **几何约束解析**：从查询解析 `(轴, 方向, rank)`，对候选框按该轴中心排序取第 rank 个——"从某端数第 k 个"变成一次确定性排序，不依赖模型是否"数对"；`closest/farthest` 改用深度图 ROI 平均深度排序，深度不可用时宁可回退也不猜；
- **枚举—选择显式化**（`detect_select.py`）：先让模型 `List EVERY <NP>` 逐个给出同类实例，再按约束取第 k 个；
- **跨查询序数链单调性**（`chains.py`）：同图、同物体、同轴同向的序数链，框中心必须随序数单调（实测 181 条有效链中 30% 破序）；对破序链做排序式重指派（L1 最优分配，另有 PAVA 保序回归变体），操作幂等。

### 3.4 多产物共识投票后处理

对主件与参照件的真分歧位（`IoU < 0.5`），由 32 套独立解码产物组成的投票池逐成员表态；当且仅当**主件零票、参照件满票**时采纳参照件。判据全部为与真值无关的纯几何量，无随机性、无自由参数；管线固定 zip 内部时间戳，**同输入必得同 SHA256**，第三方可逐条复算。

### 3.5 出件关键链路（本仓库根目录）

`_fuse_qwen.py`（判据唯一实现）→ `_enum_merge.py`（枚举合并）→ `_assembly_32b.py`（六桶互斥装配）→ `_validate_semi.py`（四项硬约束校验：条数 / 键集 / 字段逐字回显 / bbox 合法性）。

## 4. 方法学纪律

本项目以"可证明、可复算"为约束组织实验（详见 `INDEX.md` §三）：

1. **判据预登记**：`_prereg_B..N.md` 与 `_frozen_criteria.md` 在结果出来之前写定判据，以 SHA256 指纹锚定、不可事后编辑；实测修订写在独立的 `_frozen_criteria__跑后实录.md`；
2. **提交件按得分归档**：同一件不得出现两份（`_sub_path.py` 保证）；
3. **环境实录**：`_env_record.txt` 记录驱动 / CUDA / 全量 pip freeze / 模型分片 SHA256；离线段可逐字节复算，推理段不保证（浮点归约顺序随硬件变化），故记录实测环境而非断言无限复现。

## 5. 仓库结构

本仓库包含**全部代码、文档、参赛材料与可视化产物**（约 3700 个文件）。仅以下内容不入库（见 `.gitignore`）：初赛数据集（12.2GB）、复赛数据集（12.2GB）、模型权重 `weights/`（7.2GB）、第三方源码 `GroundingDINO/`、`__pycache__`。

| 目录/文件组 | 说明 |
|---|---|
| `技术文档.md` | 完整技术方案：任务、数据、模型、推理、后处理、运行配置与合规要点 |
| `INDEX.md` | 工作区目录索引：各区职责、活跃文件分组与操作纪律 |
| `消融实验复盘.md` | 消融实验记录与结论 |
| `_frozen_criteria.md` / `_frozen_criteria__跑后实录.md` | 预登记判据（指纹锚定）与跑后实录 |
| `_prereg_B..N.md` | 各实验件的判据预登记文档 |
| `_fuse_qwen.py` | 判据唯一实现，被多个装配脚本 import |
| `_enum_merge.py`、`_assembly_32b.py`、`_validate_semi.py`、`_sub_path.py` | 复赛出件关键链路：枚举合并 → 装配 → 校验 → 归档路径 |
| `_variant_det_semi.py`、`_predict_gd_3modal.py`、`predict.py`、`predict_gd_3modal.py` | 推理入口与变体探测 |
| `models/`（Python 源码） | 编码器、模态融合、GroundingDINO+LoRA、姿态分类等模型代码 |
| `config.py`、`prepare_data.py`、`evaluate.py` | 配置、数据准备与评测工具 |
| `train.py`、`train_competition.py`、`train_lora_gd.py`、`train_pose_classifier.py` | 早期训练/微调探索脚本（最终方案为零样本，未采用） |
| `requirements.txt` | Python 依赖 |
| `复赛提交材料/` | 交付包：`code/`、`result/`、README、requirements、复现流程；自包含、可独立评审 |
| `复赛提交材料文件夹/`、`答辩材料/` | 官方交付文件夹与答辩材料（技术方案 PDF、模型文件说明、代码与数据、结果文档） |
| `质询会议补充材料-赛道一熬夜炼丹师沈嘉宝/` | 质询答辩补充：模型迭代流程、消融复盘、迭代使用模型与提示词等 |
| `submission_zips/semifinal/` | 提交件档案，按实际得分归子目录（`0.4378/` … `0.7963/`、`未提交/`），扁平目录只留 `SHA256.txt` |
| `semifinal_remote_archive/`、`_srv_backup/` | 远端推理脚本与产物归档（`root/` 为服务器 `/root` 镜像） |
| `viz*`、`pick_viz`、`q38_viz_100`、`k_*_viz`、`m_viz_*` 等 | 各阶段目检图（决策类图按真实坐标绘制） |
| `_deprecated/` | 历史遗留归档区（2026-09-27 整理，无引用文件移入，含取回脚本） |
| `自证文档_result_semi_q38_nonord_来源溯源.md` | 交付件产出服务器的来源溯源自证（含逐环节哈希对拍） |

其余以 `_` 开头的根级脚本为各阶段的探测、复算与校验工具（远端通道 `_rc.py`、共识系列 `_consensus_*.py`、序数核系列 `_ordinal_cores_*.json` 等），均可独立运行。

## 6. 环境与运行

| 项 | 要求 |
|---|---|
| Python | 3.12 |
| CUDA | 12.4 |
| 离线段（装配/规则层/校验） | 纯 CPU 即可 |
| 推理段（8B / 32B-AWQ） | GPU 显存 ≥ 40GB |
| 推理段（交付件 0.7963，Qwen3.8-27B） | **2×40GB**（vLLM TP=2）或单卡 ≥ 60GB（实测 RTX PRO 6000 Blackwell 96GB） |
| 关键依赖 | torch≥2.6、transformers≥5.18（离线链路需 `transformers < 5.0` 口径见交付包）、accelerate、opencv-python、Pillow、numpy、PyYAML、tqdm；交付件级推理需 vLLM ≥ 0.22.1 |

```bash
pip install -r requirements.txt
```

模型权重与数据集未随仓库分发：

- 模型：`modelscope snapshot_download('Qwen/Qwen3.8-27B', cache_dir='./models')`，逐分片 SHA256 校验见交付包 `download_models.py --verify`；
- 数据：赛事官方初赛/复赛数据集，放置为 `data.root` 下 `Images/visible|infrared|depth` 与 `queries.json`。

复现入口（完整命令见 `复赛提交材料/复现流程.md` 与 `技术文档.md` §5）：

```bash
python scripts/infer.py --config configs/server_32b.yaml   # 端到端推理（步骤 1-5）
python scripts/consensus_vote.py                            # 共识投票后处理，产出 result.zip

# 离线复算交付件（无需 GPU，产物随仓库）：
python code/_assembly_32b.py --direct _qwen38_direct.json \
    --base result_semi_n32bf16_nonord --base-score 0.7786 \
    --ord-correct 689 --base-correct 3741 --stem result_semi_q38_nonord --zip
python code/_validate_semi.py <结果 zip>                    # 四项硬约束校验
python code/_test_assembly.py                               # 装配器夹具回归（8 组夹具/20 项断言）
```

复现判定标准：总分与 0.7963 差 ±0.005 以内，且各级净成功率落在 10%~19% 区间。

## 7. 合规声明

- 仅使用开源模型（Qwen3.8-27B 等），本地部署推理，全程无闭源在线 API 调用；
- 仅使用赛事官方测试集，未使用外部数据、未做任何人工标注；
- 所有结果由算法推理产出，后处理为确定性规则，可由管线逐条复算；
- 文档中出现的对照数字（如"对方 0.8092"）为对表实验留档，**非本队成绩**；本队交付件成绩 0.7963 全部由本队代码产出。

# 基于大模型的多模态视觉理解与推理

> **全球校园人工智能算法精英大赛 · 算法挑战赛 —— 多模态视觉定位（Visual Grounding）方案**

本项目实现从**三模态（可见光 RGB / 红外 / 深度）+ 英文文本查询**到**目标归一化边界框**的视觉定位系统。包含两套可运行管线：

| 方案 | 说明 | 状态 |
| --- | --- | --- |
| **方案 A：Grounding DINO 三模态早期融合** | 开放词汇检测器，5 通道早期融合 + 空间语义后处理 | ✅ 主推 / 提交方案，已跑通全量 |
| **方案 B：ResNet + DistilBERT 自训练管线** | 自研跨模态 Transformer 检测器，可端到端训练 | ✅ 已实现 |

---

## 一、任务与约定

### 任务定义

- **输入**：可见光 RGB + 红外（灰度）+ 深度（16-bit 单通道，单位 mm）+ 英文文本查询（query）
- **输出**：目标在可见光图像中的归一化边界框 `[x1, y1, x2, y2]`，坐标范围 `[0, 1]`
- **评估指标**：`ACC@0.5 = (IoU ≥ 0.5 的样本数) / 总样本数`
- **提交格式**：以 **Query ID 为键** 的 JSON，保留 `visible / infrared / depth / query` 字段并新增 `bbox`

### 数据约束

- 禁止调用商业闭源大模型在线 API
- 禁止手工标注；仅可使用赛方提供的测试集做推理
- 推荐开源大模型（CLIP / DistilBERT / Qwen-VL / InternVL / SAM / GroundingDINO 等）

### 核心难点

Grounding DINO 等开放词汇检测器擅长「**类别**」定位，但难以区分**同类别内的空间方位（左/右）、远近（最近/最远）与位姿（正/侧）**，会收敛到单一高置信框。本项目的核心贡献是：融合**深度与位置信息**做**空间语义后处理**，把多实例类别（斑马、人、路灯、篮球等）按语义挑出正确个体。

---

## 二、方案 A：技术路线与核心原理（主推）

> 对应 [a.txt](a.txt) 的方法路线，为当前全量推理与提交所采用。

### 2.1 架构总览

方案 A 复用 Grounding DINO（SwinT 视觉骨干 + BERT 文本编码器 + 跨模态 Decoder + 检测头），在其之上做两类关键改造：

1. **输入层多模态早期融合**：把视觉输入从 3 通道扩展为 5 通道。
2. **输出端空间语义后处理**：把「类别检测」升级为「类别 + 方位/远近/序数」的实例选择。

```
                 ┌────────────── 数据流 ──────────────┐
  RGB (3通道) ──┐                                        ┌─ 文本查询
  红外 (1通道) ──┼─ 统一缩放/归一化 ──► cat ──► (5,H,W) ──┤
  深度 (1通道) ──┘            │                          │
                             ▼                          ▼
              modify_model_first_conv: SwinT patch_embed.proj 3→5
                             │                          │
                             ▼                          ▼
                  [SwinT 视觉骨干]                [BERT 文本编码器]
                             │                          │
                             └────► [跨模态融合 Decoder] ◄┘
                                          │
                                          ▼
                              [检测头] pred_boxes + pred_logits
                                          │
                                          ▼
                        候选框 (cxcywh) + 置信度 (sigmoid)
                                          │
                                          ▼
                     parse_spatial_criterion 解析空间词
                                          │
              ┌───────────────────────────┴────────────────────────────┐
              ▼                                                        ▼
   extract_category 提炼短类别词(车辆短语/人/上位词)         颜色词? parse_target_color
   二次检测扩池 → 候选实例入池                         (人+车 query 优先车辆色)
   高瘦合并框处理: expand_candidates /                          │
   split_merged_by_depth / filter_tall_boxes                    ▼
              │                              select_color_then_spatial:
              ▼                             先颜色过滤(占比≥0.05)+面积,再空间/置信排序
   spatial_select: NMS → 面积过滤 → 按深度/横/纵排序取第 rank 个
                                          │
                                          ▼
                         to_xyxy_norm → sanitize_box → [x1,y1,x2,y2]∈[0,1]
                                          │
                                          ▼
                    result_gd_3modal_v2.json (QueryID 索引，支持断点续跑)
```

### 2.2 关键技术组件

| 组件 | 位置 | 职责 |
| --- | --- | --- |
| `load_multimodal_image` | [predict_gd_3modal.py](predict_gd_3modal.py) | 加载三模态并统一尺寸，拼接为 5 通道张量 |
| `parse_spatial_criterion` | [predict_gd_3modal.py](predict_gd_3modal.py) | 空间词解析，返回 `(key, order, rank)` |
| `extract_category` | [predict_gd_3modal.py](predict_gd_3modal.py) | 提炼短类别词（车辆短语 / person / 上位词），供二次检测扩池 |
| `parse_target_color` / `parse_vehicle_color` | [predict_gd_3modal.py](predict_gd_3modal.py) | 解析目标颜色；人+车混合 query 优先取车辆颜色 |
| `_color_mask` / `box_color_score` | [predict_gd_3modal.py](predict_gd_3modal.py) | HSV 颜色掩码与框内颜色占比打分 |
| `select_color_then_spatial` | [predict_gd_3modal.py](predict_gd_3modal.py) | 颜色+空间可组合：先颜色/面积过滤，再子集内空间或置信度排序 |
| `spatial_select` | [predict_gd_3modal.py](predict_gd_3modal.py) | NMS 去重 + 面积过滤 + 空间排序选优 |
| `expand_candidates` / `split_merged_by_depth` / `filter_tall_boxes` | [predict_gd_3modal.py](predict_gd_3modal.py) | 高瘦合并框：上位词重检测、深度切分、非竖向实体时剔除 |
| `to_xyxy_norm` / `sanitize_box` | [predict_gd_3modal.py](predict_gd_3modal.py) | 坐标格式转换与合法性保证 |

### 2.3 核心实现机制

#### (1) 三模态早期融合（输入层）

- **通道拼接**：RGB(3) + 红外(1) + 深度(1) 统一 `RandomResize([IMAGE_REF], max_size=1333)` 后按通道 `cat` 为 `(5, H, W)`。
- **红外处理**：三通道转单通道（灰度）。
- **深度处理**：16-bit（mm）经 `clip[300, 19999]` → 线性归一化到 `[0,1]`；`0` 或越界像素视为无效，用有效区域均值填充，避免深度缺失造成排序错误。
- **权重初始化**：SwinT `patch_embed.proj`（`backbone[0]` 处）改为 5 通道卷积，前 3 通道复制预训练 RGB 权重，后 2 通道 `std=0.01` 高斯初始化——既复用预训练表达，又让新模态从零学习。

#### (2) 空间语义增强（输出端后处理）

这是针对「同类别实例区分」的核心提分模块，工作流如下：

1. **空间词解析** `parse_spatial_criterion`：
   - 归一化连字符/下划线（`far-right` → `far right`）。
   - 解析**序数** `first/second/.../last` 得到 `rank`（1/2/…/-1）。
   - 判定排序键与方向，优先级 **深度（远/近）> 水平（左/右）> 垂直（上/下）**，返回 `(key, order, rank)`。
2. **短类别提炼** `extract_category`：`strip_to_class` 对带大量定语/从句的句子几乎删不掉内容（如「person in a pink shirt driving a red bumper car on the left background」仍是整句），二次检测因此无新增实例。`extract_category` 改为：句子含车辆名词 → 返回车辆短语（`bumper car`），主语为人 → 返回 `person`，否则回退上位词/`strip_to_class`。
3. **候选收集阈值**：`CANDIDATE_THRESHOLD=0.15`。边缘实例（如远左车辆）的 logit 常贴在 0.2 附近，跨进程/并发推理时数值轻微浮动会使其掉出候选池；0.15 留出裕量，主流程固定 `torch/numpy` 种子保证可复现。
4. **二次检测扩池**：用提炼出的短类别词对同一张 5 通道图再推理，把左/右/远处同类实例补入候选池（单次全 query 检测常漏掉侧边实例）。
5. **高瘦合并框处理**：对疑似「人+前景物/多实例」纵向合并框，用上位词重检测（`expand_candidates`）、按深度不连续处切分（`split_merged_by_depth`）等拆出子候选；存在非高瘦候选时剔除高瘦合并框（`filter_tall_boxes`），竖向实体（人/树等）除外。
6. **颜色属性匹配**：
   - `parse_target_color` 解析目标颜色；「person … driving a red car」类 query 优先取**车辆颜色**，避免误取衬衫色。
   - `_color_mask`/`box_color_score` 用 HSV 掩码计算框内颜色像素占比。
   - `select_color_then_spatial`：颜色与空间**可组合**——先按颜色占比 ≥0.05 且面积 ≥0.005 过滤出子集，再在子集内做空间/置信度排序；颜色全不匹配时交回全量空间排序兜底。
7. **空间排序选优** `spatial_select`：
   - `torchvision.ops.nms(boxes, logits, iou=0.3)` 去重，保留相邻独立实例。
   - **面积过滤**：剔除面积占比 > `MAX_BOX_AREA_FRAC(0.35)` 的整图合并大框，避免「最近/最远」误选到整排目标的包络框。
   - 按空间键排序：`x` 用框中心横坐标、`y` 用纵坐标；`depth` 用 `depth_at_box`（框内有效深度 p=50 分位数，最远取大、最近取小）。
   - 取排序后第 `rank` 个候选；`rank=-1` 取末尾。

#### (3) 输出合规

- 检测头原始输出为 `(cx, cy, w, h)` 归一化中心宽高框。
- `to_xyxy_norm` 转 `[x1,y1,x2,y2]` 并 `clamp[0,1]`。
- `sanitize_box` 保证 `x1<x2 && y1<y2` 且范围合法。
- 无有效候选时兜底返回 `[0.25, 0.25, 0.75, 0.75]`（并标记 `error` 字段以防隐藏失败）。

### 2.4 兼容性补丁

针对在线环境库版本与沙箱限制，对模型源码打了 3 处补丁：

| 文件 | 补丁 | 目的 |
| --- | --- | --- |
| `groundingdino/util/slconfig.py` | 移除模块级 `yapf` 导入 | 绕开沙箱 `import yapf` 卡死 |
| `groundingdino/models/GroundingDINO/bertwarper.py` | 内联 `get_head_mask` / `get_extended_attention_mask` | 兼容新版本 transformers 移除的方法 |
| `groundingdino/models/GroundingDINO/ms_deform_attn.py` | 无编译 C++ 扩展时回退纯 PyTorch | 无 CUDA 算子也能运行 |

---

## 三、方案 B：ResNet + DistilBERT 自训练管线

> 自研可微训练检测器，用于对比 / 后续微调基座。

### 架构

1. **文本编码器** `models/encoders.py::TextEncoder`：基于 `distilbert-base-uncased`，输出序列特征 + 加权平均池化。
2. **视觉多模态编码器** `MultiModalImageEncoder`：RGB / IR / Depth 三模态共享 backbone（默认 timm `resnet18`），独立权重；提取 layer4 空间 + 全局特征投影后做**门控聚合**（softmax 加权和）。
3. **跨模态融合** `models/fusion.py::CrossModalFusion`：文本为 query、视觉 spatial 为 key/value，Pre-LN Transformer Decoder。
4. **BBox 头** `BBoxHead`：2 层 MLP → `(cx, cy, w, h)`，sigmoid 映射到 `[0,1]`。

### 损失与预处理

- 损失：`loss = λ_l1·L1 + λ_iou·(1−IoU) + λ_giou·(1−GIoU)`
- 预处理 `utils/preprocess.py`：可见光 letterbox 等比缩放 + ImageNet 归一化；红外取单通道复制 3 通道；深度 clip `[300,19999]` 线性归一化。

---

## 四、目录结构

```
2026AiC/
├── a.txt                          # 方法路线说明（赛题指南）
├── config.py                      # 全局配置（路径/超参/模型选项）
├── requirements.txt               # Python 依赖
│
│  ── 方案 A（Grounding DINO 主推）──
├── predict_gd_3modal.py           # 三模态早期融合 + 空间/颜色语义增强推理主脚本
├── result_gd_3modal_v2.json       # 当前全量推理输出（QueryID → 含 bbox，支持断点续跑）
├── result_gd_3modal.json          # 历史输出（旧版逻辑，已停用）
├── weights/
│   ├── groundingdino_swint_ogc.pth            # GroundingDINO 官方权重
│   ├── gdino_ms/AI-ModelScope/GroundingDINO/  # 模型源码（含兼容补丁）
│   └── bert/bert-base-uncased/                # 本地 BERT tokenizer
│
│  ── 方案 B（自训练管线）──
├── prepare_data.py                # 数据校验 & train/val 划分
├── train.py                       # 训练入口
├── evaluate.py                    # 计算 ACC@0.5、mIoU
├── predict.py                     # 测试集推理 & 生成提交 JSON/zip
├── datasets/mmvg_dataset.py       # Dataset & collate
├── models/{encoders,fusion,grounding_model}.py
├── losses/box_loss.py             # L1+IoU+GIoU
├── checkpoints/                   # 训练产出（best.pth / latest.pth / train.log）
│
│  ── 数据与可视化 ──
├── 初赛数据集-基于大模型的多模态视觉理解与推理/
│   ├── test.json
│   └── Images/{visible,infrared,depth}/*.png
├── viz_out/                       # 单样本核对图（{qid}_final.png 仅最终红框 / _pool.png 候选池）
├── viz_v2_check/                  # 按场景分组的多 query 抽查图（viz_partial_check.py 生成）
├── detect_viz.py / viz_boxes.py   # 检测+可视化工具（与主流程选择逻辑一致）
└── result.json / result_fixed.json / result.zip   # 历史提交产物（result.zip 为旧可提交包）
```

---

## 五、环境与依赖

```bash
pip install -r requirements.txt
```

- Python ≥ 3.9（建议 3.8–3.12）
- PyTorch ≥ 2.0（已测 2.5.1+cu121；CPU 也可运行但较慢）
- `transformers`、`torchvision`、`opencv-python`、`numpy`、`Pillow`
- 方案 A 权重：`weights/groundingdino_swint_ogc.pth`
- 方案 B 权重：`checkpoints/exp/best.pth`（由训练产出）

> 注：方案 A 已本地化模型源码与 BERT tokenizer，可离线推理（`HF_HUB_OFFLINE=1`）。

---

## 六、使用方法

### 方案 A：生成结果（当前主推，含空间语义增强）

```bash
# 全量推理（默认分辨率 800，当前主用；建议 python -u 便于实时看日志）
python -u "d:\Desktop\2026AiC\predict_gd_3modal.py" \
    --out "d:\Desktop\2026AiC\result_gd_3modal_v2.json"

# 降分辨率加速（越小越快；实测 560 约 0.5 img/s）
python -u "d:\Desktop\2026AiC\predict_gd_3modal.py" --ref 560 \
    --out "d:\Desktop\2026AiC\result_gd_3modal_v2.json"

# GPU 上 FP16 推理加速
python -u "d:\Desktop\2026AiC\predict_gd_3modal.py" --half --ref 560 \
    --out "d:\Desktop\2026AiC\result_gd_3modal_v2.json"
```

断点续跑与保护：

- 输出文件若已存在，自动跳过其中已完成的 query，从断点继续（`--n` 调试模式除外）。
- 每 100 条或每 10 分钟自动写一次 checkpoint，中断后直接重跑同命令即可续跑。
- 主流程入口固定 `torch/numpy` 随机种子（seed=0），配合 `CANDIDATE_THRESHOLD=0.15` 使边缘候选稳定入池、结果可复现。

**最终提交包**：`result_gd_3modal_r1pp2.zip`（内含 `result_gd_3modal.json`，9555 条；生成逻辑与校验见第七节）。

参数：

- `--n N`：仅处理前 N 条（调试）
- `--ref N`：输入图像参考边（越小越快，默认 800）
- `--half`：GPU 上 FP16 加速
- `--out PATH`：输出 JSON 路径

### 方案 B：自训练流程

```bash
python prepare_data.py
python train.py --epochs 30 --batch_size 8 --lr 1e-4
python evaluate.py --ckpt checkpoints/exp/best.pth --val_json val.json
python predict.py --ckpt checkpoints/exp/best.pth --test_json test.json \
    --output result.json --save_zip
```

### 抽样可视化（核对框位）

```bash
python "d:\Desktop\2026AiC\_viz_6_spatial.py"    # 6 张场景增强后重绘
python "d:\Desktop\2026AiC\_table_6.py"          # 逐 query 明细表
```

---

## 七、验证结果

> 全量推理已在 A100 服务器上用本仓库同一份推理代码跑完（9555/9555，耗时约 170 分钟），输出 `result_gd_3modal_v2.json`（0 error、bbox 全部合法、key 与测试集完全对齐）；下列为人工核对确认过的抽样结论。

- **云端/本地一致性**：两地同配置结果在交集样本上一致（~1e-3 以内差异占 99%+），仅极少数样本因数值浮动出现实例翻转；最终以**云端结果**为基线。

- **最终提交（本轮）**：`result_gd_3modal_r1pp2.zip`（内含 `result_gd_3modal.json`，9555 条）
  - 生成逻辑（保守融合 + 四道护栏）：
    1. 基线：云端三模态 GD 结果；
    2. **G4 替换**：仅当 LocateAnything-v2 与 GroundingDINO 互相一致（IoU≥0.5）且基线孤立（IoU<0.5）时替换为 `0.6·Locate + 0.4·GroundingDINO`，且替换框须 **面积 ≤0.35 且 ≤1.25×基线**；
    3. **R1′（序数护栏）**：含序数词且基线面积 >0.2 的 **75 条**改用 Locate 框（基线在序数类常输出同一大框）；
    4. **R2（集合护栏）**：集合/复数类 **禁止缩小替换**（**21 条**回退较大框）；
    5. **R1″（空间大框护栏，精炼版）**：纯空间词（非序数）且基线面积 >0.35、**且 Locate 框为物体尺度（≤0.35）**的 **128 条**改用 Locate 框；Locate 同样给出大区域的 **47 条保留基线**（这类 query 目标本身即大区域：land / water / fence / walkway / grass，替换无益甚至有害，如 `016606_001` M=0.733→L2=0.997）
  - 相对基线共改动 **588 条（6.2%）**
  - 校验：0 字段异常 / 0 非法 bbox / key 与测试集完全对齐；4 个人工确认样本**零改动**（IoU = 1.000 vs 基线）；大框（面积>0.35）**385 → 230**
  - SHA256(zip) = `B87B9985C14E9D39609A689F780D81AF1B5215F17E68504EAE9148A21D45A292`
  - 回退：`_deprecated/` 保留 `result_gd_3modal_g4_r2r1p_r1pp.zip`、`..._g4_r2r1p.zip`、`..._g4.zip`、`..._guard866.zip`、`..._conservative_885.zip`

- **颜色 + 空间组合修复（`000282` 碰碰车场景，ref 800 人工确认）**：
  - `000282_004`（green and white bumper car parked empty on the left background）：旧逻辑颜色/空间互斥，排序选到左前黄色车；改为颜色过滤（green）+ 左排后正确选中左背景绿白车 `[0.20,0.30,0.31,0.40]`。
  - `000282_008`（person in a pink shirt driving a red bumper car on the left background）：旧候选池缺远左红车（二次检测用整句无新增实例）；改用 `extract_category` 后远左红车入池，red 过滤 + 左排选中 `[0.03,0.32,0.15,0.51]`（粉衬衫司机），而非中部黑衬衫司机红车。
- **颜色属性（历史抽样）**：`002004_001` 黄衣玩偶正确选中 `[0.421,0.291,0.469,0.436]`。
- **空间增强生效**：
  - `017005` 三只斑马：按「右起第一只 / 左起第一只 / 最远」正确分离为三个不同框。
  - `010705`：`far-right person` 与 `second person on the right` 为两个不同框（序数修复）。
  - `012532` 四棵树：按「从右到左第 1/2/3/4 棵」x 坐标严格递减排列。
- 抽样可视化：`viz_out/{qid}_final.png`（仅最终红框）、`viz_v2_check/{scene}_v2.png`（整场景多 query 对照）。

---

## 八、已知局限

1. **合并类目标无法拆分**：检测器把整排同类别（如远处停放的多辆车 `010572`）合并为单框，空间后处理无法再切分。
2. **正/侧面目标区分受限**：`002004` 两个玩偶（一正一侧），检测器只输出「单个玩偶 + 人/玩偶合并框」；且该场景红外亮度与深度中玩偶与人粘连成片、不可分，无法用红外面积区分正/侧。
   - 以上两者同属「**检测端多实例分离 + 位姿识别**」能力不足，需依赖 LoRA 微调。
3. **速度瓶颈**：`ms_deform_attn` 无编译 C++ 扩展回落纯 PyTorch。本机 MX550（2GB）`ref 800` 实测约 `0.22 img/s`；A100-40GB 上约 `0.94 img/s`，全量 9555 条约 170 分钟。已实现 checkpoint 断点续跑缓解中断风险。

---

## 九、技术优化方向

> 按「预期收益 / 可行性」排序，每项给出**可行性、预期效果、实施步骤**三要素。

### 9.1 P0 —— 检测端多实例分离（LoRA 微调）

直接针对已知局限 1、2，是当前分数提升的最大杠杆。

- **可行性**：中高。Grounding DINO 骨干与文本编码器均可注入 LoRA，训练量小（约 <1% 参数量）。
- **预期效果**：解决「多辆车合并」「正/侧玩偶」等整排/位姿问题，对被合并类别样本的 IoU 提升幅度最大。
- **实施步骤**：
  1. 用赛方可提供的训练集（或合成增强数据）构造 `(5通道图, 文本, 单bbox)` 监督样本。
  2. 在 SwinT 层与 BERT 层注入 LoRA 适配器，冻结基座。
  3. 以 L1 + GIoU 损失微调，恢复时替换 `predict_gd_3modal.py` 的权重加载逻辑。

### 9.2 P0 —— 特征层融合（替换输入拼接）

- **可行性**：高。在 SwinT 输出后的多尺度特征处插入融合模块，改动可控。
- **预期效果**：深度/红外特征以可学习方式作用于定位，尤其改善「远近」判定，优于直接通道拼接。
- **实施步骤**：
  1. 分别过三个独立 patch_embed 分支（或共享骨干 + 门控），得到 RGB/IR/Depth 特征。
  2. 设计门控或交叉注意力融合模块（可复用方案 B 的 `CrossModalFusion`）。
  3. 拼接后送入 Decoder，微调收敛后验证 ACC@0.5。

### 9.3 P1 —— 空间语义规则扩展

- **可行性**：高（纯规则，无训练成本）。
- **预期效果**：边际提升，减少规则漏判（覆盖 `between` / `behind` / `in front of` 等介词短语）。
- **实施步骤**：
  1. 扩充 `parse_spatial_criterion` 关键词表，覆盖更多方位表达。
  2. 增加「深度分桶 + 相对位置」的复合判定规则。
  3. 在 `_viz_full` 抽样集上回归验证无副作用。

### 9.4 P1 —— 深度预处理与排序优化

- **可行性**：高。
- **预期效果**：改善深度排序稳定性，减少无效深度导致的选优退化。
- **实施步骤**：
  1. 试用对数压缩（`log1p`）替代线性归一化，保留近场细节。
  2. `depth_at_box` 引入边缘保持或多尺度有效区域采样。
  3. 对比 `_normalize_depth` 各方案在抽样集上的选优正确率。

### 9.5 P2 —— 推理加速（性能）

- **可行性**：中（需 CUDA 编译环境）。
- **预期效果**：速度 5–10×，把全量回放从「小时级」降到「分钟级」。
- **实施步骤**：
  1. 编译 `ms_deform_attn` 的 CUDA C++ 扩展（当前因缺编译环境回落纯 PyTorch）。
  2. 或导出 ONNX → TensorRT 做 FP16/INT8 推理。
  3. 增加 batch 推理，摊薄 Python 侧开销。

### 9.6 P2 —— 多模型集成

- **可行性**：中。
- **预期效果**：融合 Grounding DINO 与其他开放词汇检测器结果，降低单模型偏差。
- **实施步骤**：
  1. 接入第二检测器（如 YOLO-World / 其他 GD 变体）。
  2. 对两模型 bbox 做 IoU 加权融合或投票。
  3. 在抽样集上对比融合前后 ACC@0.5。

---

## 十、复现性

- 方案 A 推理入口固定 `torch.manual_seed(0)` / `np.random.seed(0)`，配合 `CANDIDATE_THRESHOLD=0.15` 稳定候选池；方案 B 固定 `config.SEED = 42`
- 训练日志、配置、权重均落到 `checkpoints/`
- 评估与预测脚本对输入数据无随机扰动

---

## 十一、关键文件速查

| 文件 | 说明 |
| --- | --- |
| [predict_gd_3modal.py](predict_gd_3modal.py) | 方案 A 主脚本：三模态融合 + 空间/颜色增强 + 断点续跑 |
| [result_gd_3modal_v2.json](result_gd_3modal_v2.json) | 当前全量输出（修复后逻辑，运行中/可分片查看） |
| [detect_viz.py](detect_viz.py) | 单样本检测+可视化（与主流程选择逻辑一致） |
| [viz_partial_check.py](viz_partial_check.py) | 读结果 JSON 按场景生成多 query 抽查图 |
| [config.py](config.py) | 全局配置 |
| [a.txt](a.txt) | 方法路线 / 赛题指南 |
| [prepare_data.py](prepare_data.py) | 数据准备 |
| [train.py](train.py) | 方案 B 训练 |
| [evaluate.py](evaluate.py) | 评估 ACC@0.5、mIoU |
| [predict.py](predict.py) | 方案 B 预测 & 提交 JSON |
# 基于大模型的多模态视觉理解与推理（Visual Grounding）

面向 **2026 AIC 人工智能算法精英大赛 · 基于大模型的多模态视觉理解与推理赛道** 的参赛工程。

核心任务：给定 **可见光（RGB）+ 红外（Infrared）+ 深度（Depth）** 三模态图像，以及一条描述目标的英文查询文本（Query），利用开源大模型完成**视觉定位（Visual Grounding）**，输出目标在可见光图像中的**归一化边界框 `[x1, y1, x2, y2]`**。

---

## 1. 技术路线

本工程以开源生成式 VLM **Qwen3-VL-8B-Instruct**（Apache-2.0，当前唯一已登记模型）为核心，构建零样本开放集视觉定位管线，并加入三模态多图输入、查询文本清洗、搜索式候选投票（可选）、序数链一致性修正（可选）等增强，提升复杂场景鲁棒性：

```
queries.json ──► 按 Query ID 关联三模态图像与文本
        │
        ▼
查询文本清洗（BOM / 零宽字符 / 连字符统一）
        │
        ▼
多模态预处理：RGB 归一化 / 红外去噪 / 深度离群剔除 /
低光红外增强（fusion_mode: low_light）或逐像素门控三模态融合（gated）
        │
        ▼
模型前向（Qwen3-VL：三模态多图对话输入 multimodal_input_mode=split，
         模式①逐查询单框 / 模式③同图多框 multi_box，带同图视觉特征缓存；
         可选 num_samples>1 候选采样 + IoU 一致性投票）
        │
        ▼
文本 ↔ 检测框匹配（精确 → 双向子集 → 顺序子串 + 空间/深度约束排序）
        │
        ▼
（可选）序数链一致性修正（chain_enforce: sort | project，跨查询单调性）
        │
        ▼
bbox 归一化 / 合法性校验
        │
        ▼
导出预测 JSON  →  zip 提交包
```

**评测指标**：`ACC@0.5`（预测框与真实框 IoU ≥ 0.5 记为正确）。

## 2. 目录结构

```
├── configs/
│   ├── config.yaml           # 全局配置（当前模型 qwen3_vl，数据根 dataset）
│   └── cloud_qwen.yaml       # 云端 Qwen3-VL 配置（三模态多图 + 模式③多框 + bf16）
├── dataset/                  # 当前生效数据集 = 复赛（Images 三模态 + queries/queries.json）
├── data/
│   ├── README.md             # 数据目录说明
│   └── test/                 # 测试集视图（Images 为指向 dataset/Images 的符号链接，不入库）
├── checkpoints/              # 模型权重目录（git 忽略，不入库；当前本地为空，Qwen3-VL 权重仅云端）
├── src/
│   ├── config.py             # 配置加载（Data/Model/Inference/Qwen/Output/Postprocess/Logging）
│   ├── data/                 # 数据解析 + 多模态预处理 + 融合（loader / preprocess / fusion）
│   ├── models/               # 模型层（base 接口 + factory 工厂 + qwen3_vl + qwen3_vl_parse）
│   ├── inference/            # 端到端推理 + 文本↔检测框匹配 + 序数链修正（predictor / matcher / chains）
│   ├── postprocess/          # bbox 处理与校验 + 候选搜索与投票聚合（bbox / search）
│   ├── eval/                 # ACC@0.5 / IoU 评测
│   └── utils/                # 日志、IO、文本清洗、LRU 缓存（text / cache / io / logging_utils）
├── scripts/                  # 命令行入口（推理/评测/导出/数据探查/序数链分析/守门/归因/快照/云端编排）
├── tests/                    # 单元测试（251 项，不依赖模型权重）
├── tools/                    # 云端部署 CLI（deploy.py / ssh_client.py）
├── docs/                     # 全部文档（方案/说明/模型 分类归档，见 §5）
└── outputs/                  # 运行产物（predictions / submission，git 忽略；archive/ 存历史运行）
```

## 3. 快速开始

### 3.1 安装依赖

```bash
# 1) 创建虚拟环境（推荐）
python -m venv .venv
# Windows 激活
.venv\Scripts\activate

# 2) 安装深度学习框架（按有无 GPU 二选一）
# CPU：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# GPU（CUDA 12.1 示例）：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3) 安装其余依赖
pip install -r requirements.txt
```

### 3.2 数据与模型准备

- 将官方数据集放在项目根目录的 `dataset/` 下（含 `Images/visible|infrared|depth` 与 `queries/queries.json`；`configs/config.yaml` 的 `data.root` 默认指向 `dataset`）。
- 将 Qwen3-VL-8B-Instruct 权重放在 `checkpoints/Qwen3-VL-8B-Instruct/`（`configs/config.yaml` 的 `model.name_or_path` 默认指向它）。权重文件 `*.safetensors` 不入库。
- Qwen3-VL-8B 推理需 ≥12GB 显存 GPU；本地无该条件时请走云端（`configs/cloud_qwen.yaml` + `tools/deploy.py`，详见 [AutoDL部署说明](./docs/说明/AutoDL部署说明.md)）。云端下载建议用 ModelScope（HF 镜像大文件易断连）。
- 数据集与模型资产的完整登记见 [docs/模型/模型清单.md](./docs/模型/模型清单.md)。

### 3.3 运行

```bash
# 冒烟验证（前 2 张图跑通全流程）
python scripts/infer.py --limit 2

# 全量推理
python scripts/infer.py

# 导出提交 zip
python scripts/export_submission.py

# 单元测试
python -m pytest tests/ -q

# 提交前守门（无真值环境）：新预测 vs 基线，判「可证明错误」净减少与收益背书比
# 注意：-p 为基线（A），-c 为对比/新预测（B），与脚本 help 一致
python scripts/复赛/provable_errors.py -p outputs/复赛/archive/run_20260924_090232_qwen_multiimg.tar.gz \
       -c <新预测.json|run_*.tar.gz> \
       --json outputs/复赛/audit/provable_errors_new.json --top 10

# 形态自检：改动是「平移错位」还是「置换重指派」（纯修复类改动应判为置换）
python scripts/复赛/chain_shift_probe.py -p <基线> -c <新预测> \
       --json outputs/复赛/audit/shift_probe_new.json
```

> 复赛测试集**无公开真值**，官方 ACC@0.5 只在提交后返回。任何改动上云前必须先过上面这道
> 守门（判据：可证明错误净减少 > 0 **且** 收益背书比 ≥ 1.0），否则等同盲改。
> 依据与完整路线见 [提分方案-推理准确度](./docs/方案/提分方案-推理准确度.md)。

```bash
# 上云前审计：量出三个代码级变量各影响多少条 query（本地，无需云端数据）
python scripts/复赛/round_audit.py --json outputs/复赛/audit/round_audit.json

# 多变量同轮的事后归因：一次全量跑多个变量，本地离线还原各变量的反事实臂
python scripts/复赛/decompose_arms.py \
       --diag outputs/复赛/archive/run_<ts>_mv6.tar.gz \
       --predictions outputs/复赛/archive/run_<ts>_mv6.tar.gz \
       --baseline outputs/复赛/archive/run_20260924_090232_qwen_multiimg.tar.gz \
       --out-dir outputs/复赛/audit/arms_mv6 --json outputs/复赛/audit/decompose_mv6.json
```

> 本轮采用**多变量同轮**策略（一次全量叠加多个变量 + 事后离线归因），
> 作业指导见 [云端运行说明书-多变量同轮](./docs/说明/云端运行说明书-多变量同轮.md)。

### 3.4 Qwen3-VL 生成式定位

当前唯一模型类型为 `qwen3_vl`（`src/models/factory.py` 的 `_REGISTRY`）。核心能力：

- **三模态多图输入**：`multimodal_input_mode=split`（默认）把可见光 + 红外 + 深度各渲染为一张图，以多图对话喂入 Qwen，bbox 落在图 1（可见光）；单图伪 RGB 融合路径（`fused`）保留为配置项。
- **两种输出模式**（`qwen.output_mode`）：`single_box`（模式①，逐查询一次生成）与 `multi_box`（模式③，同图一次生成多框，生成次数从"每查询一次"降到"每图一次"；默认）。
- **ID 化输出**（`qwen.multi_box_id`，默认 `false`）：09-30 全量验证后回退——ID 化改 prompt 引发 4276 条输出漂移、官方 ACC 0.7292→0.7230 负收益（详见 [云端全量推理记录-20260930](./docs/说明/云端全量推理记录-20260930.md)）；现恢复 09-24 基线整句 label 回显。置 `true` 可复现 ID 化行为做 A/B。
- **同图视觉特征缓存**：`feature_cache=true`（默认）以图像内容哈希为键缓存视觉塔输出，同图 N 条查询只编码一次，结果不变。
- **搜索式推理**（可选）：`qwen.num_samples>1` 时对同一 query 温度采样多个候选，按 IoU 一致性投票聚合（`src/postprocess/search.py`）。
- 输出坐标为 **[0,1000] 归一化**，解析与换算见 `src/models/qwen3_vl_parse.py`（纯函数，可单测）。

接入与解析细节见 [模型适配方案-当前代码接入Qwen](./docs/方案/模型适配方案-当前代码接入Qwen.md)，云端全量运行见 [云端全量推理记录-20260924](./docs/说明/云端全量推理记录-20260924.md)。

## 4. 更换模型

模型层采用「抽象基类 + 工厂」设计，换模型不碰推理主流程：

1. 新建一个继承 `src/models/base.py::BaseGroundingModel` 的封装类（实现 `predict_boxes`）；
2. 在 `src/models/factory.py` 的 `_REGISTRY` 中登记一行（当前已登记：`qwen3_vl`）；
3. 修改 `configs/config.yaml`：

```yaml
model:
  type: "你的模型类型名"          # 对应 factory.py 登记键
  name_or_path: "..."          # 本地目录名 / 绝对路径 / HF 在线 ID（含 /）
```

推理脚本无需改动，`run_inference()` 会自动按 `model.type` 创建对应模型。

> 更换模型后请同步更新 [docs/模型/模型清单.md](./docs/模型/模型清单.md)，并在运行记录中**记录结果使用的模型**（长期约定；流程详见 [docs/模型/模型更换说明.md](./docs/模型/模型更换说明.md)）。
>
> **结果保存约定**（参考 `outputs/复赛/archive/run_20260924_090232_qwen_multiimg/`）：每次云端全量推理结果按 `outputs/<split>/archive/run_<时间戳>_<描述>/` 归档，须含 `predictions.json`、`submission.json`、`submission(<成绩>).zip`、`run.log`、环境/代码/配置三快照与 `MD5SUMS.txt`，并打包回传。

## 5. 主要文档

- [代码文件功能说明](./docs/说明/代码文件功能说明.md)：**全部代码文件**（src / scripts / tests / tools / 根级文件）的功能速览。
- [模块说明](./docs/说明/模块说明.md)：src 各模块职责、接口与数据流。
- [项目完整说明](./docs/说明/项目完整说明.md)：工程全貌（架构 / 数据 / 模型 / 推理 / 运行）。
- [当前模型使用与环境介绍](./docs/当前模型使用与环境介绍.md)：当前生效模型、推理管线、云端/本地环境与运行记录。
- [数据处理与模型输入输出说明](./docs/说明/数据处理与模型输入输出说明.md)：以数据流为主线，说明处理、输入输出构造与解析、后处理与导出。
- [模型适配方案-当前代码接入Qwen](./docs/方案/模型适配方案-当前代码接入Qwen.md)：Qwen3-VL 生成式定位的代码接入方案（输出解析 / 坐标换算 / 缓存）。
- [Qwen3-VL适配性审查报告](./docs/方案/Qwen3-VL适配性审查报告.md)：Qwen3-VL 适配性审查结论（坐标体系 / transformers 版本 / 显存）。
- [数据集说明](./docs/dataset-guide/数据集说明.md)：复赛数据集全量探查结论（含统计图 `data_overview.html`）。
- [四步解题思路-方法说明](./docs/方案/四步解题思路-方法说明.md)：按赛题四步解题思路逐项对应代码实现。
- [AutoDL 部署说明](./docs/说明/AutoDL部署说明.md)：把工程部署到 AutoDL 云 GPU 的完整流程。
- [云端运行说明书](./docs/说明/云端运行说明书.md)：**照做即可**的上云作业指导（配置核对 → 建实例 → 装依赖/下权重 → 冒烟 → 全量 → 回传 → 七步校验），含本轮改动说明与决策规则。
- [云端运行说明书-多变量同轮](./docs/说明/云端运行说明书-多变量同轮.md)：**本轮推荐入口**——一次全量叠加多个变量，并把每个变量的反事实证据留在诊断里，回传后本地离线还原各臂完成逐变量归因（省下多轮全量）。
- [复赛离线回归报告](./docs/方案/复赛离线回归报告.md)：无云端推理的代理指标回归（基线口径、占位/共框归因、token 预算定案 §5.6、链修正裁决 §8）。
- [提分方案-推理准确度](./docs/方案/提分方案-推理准确度.md)：无真值下的提分路线——**可证明错误面板与改动守门器**、P0/P1/P2 分级动作、已证伪项与红线。
- [云端全量推理记录-20260924](./docs/说明/云端全量推理记录-20260924.md)：复赛 Qwen3-VL 云端全量运行记录（Base 轮见 20260923）。
- [云端全量推理记录-20260928](./docs/说明/云端全量推理记录-20260928.md)：带 `diag_dump` 的诊断轮，定案丢框成因（89/92 为截断、`echo_fail=0`）。
- [模型更换说明](./docs/模型/模型更换说明.md)：如何更换/新增视觉定位模型（含登记规范）。
- [模型清单](./docs/模型/模型清单.md)：已使用模型资产登记（`checkpoints/` 目录对应关系）。
- [技术方案](./docs/方案/技术方案.md)：算法设计思路与实现细节。
- [使用说明](./docs/说明/使用说明.md)：环境搭建与运行细节、常见问题。
- [技术报告大纲](./docs/方案/技术报告大纲.md)：复赛/半决赛技术报告撰写框架。

## 6. 赛题合规说明

- 仅使用**开源大模型**（Qwen3-VL-8B-Instruct，Apache-2.0；历史轮次使用的 Grounding DINO / OWLv2 亦为开源），不调用商业闭源在线推理 API。
- 仅使用官方**测试数据**（当前为复赛数据集）做**零样本推理，不训练、不伪标注**，不带微调。
- 所有预测由算法模型自主推理输出，无人工干预。
- 预测结果严格按赛题格式输出，不修改除 `bbox` 外的其它字段。

# Multimodal Visual Grounding（基于大模型的多模态视觉理解与推理）

> 全球校园人工智能算法精英大赛（AIC 2026）· 赛道一：基于大模型的多模态视觉理解与推理
> 队伍：熬夜炼丹师（AIC-2026-14375345）

## 任务定义

给定同一场景的**可见光（visible）、红外（infrared）、深度（depth）**三模态图像与一条英文查询，输出目标在可见光图像中的归一化边界框 `[x1, y1, x2, y2]`。评测指标为 **ACC@0.5**（IoU ≥ 0.5 记为正确）。

复赛数据规模：1295 张场景图 / 5690 条查询；约 19.4% 的查询含序数指代（如 "the third one from the right"）。

## 方案路线

采用**预训练多模态大语言模型零样本推理 + 确定性规则层**的路线：

1. **三模态预处理**：可见光归一化；红外转灰度、中值去噪；深度剔除无效区并保留掩码；弱光场景提供红外加权融合与逐像素门控融合两条增强路径；
2. **大模型推理**：Qwen3.8-27B-Instruct（bfloat16，本地部署）以 `multi_box` 模式一次前向生成多框，配合按图像内容哈希的视觉特征 LRU 缓存；
3. **候选匹配与序数层**：三级匹配链（精确 → 双向子集 → 顺序子串）+ 空间/序数约束的规则排序，把模型不擅长的计数与排序交给纯几何规则；
4. **多产物共识投票后处理**：对真分歧位做满票采纳的确定性融合。

线上最高得分：**0.7598**（`result_semi_enum_trunc.zip`）。

## 仓库内容

本仓库仅包含**代码与文档**。数据集（约 25GB）、模型权重（约 7.5GB）、可视化输出与 GroundingDINO 第三方源码均不入库（见 `.gitignore` 白名单规则）。

| 文件/组 | 说明 |
|---|---|
| `技术文档.md` | 完整技术方案：任务、数据、模型、推理、后处理与合规要点 |
| `INDEX.md` | 工作区目录索引：各目录职责、活跃文件分组与操作纪律 |
| `消融实验复盘.md` | 消融实验记录与结论 |
| `_frozen_criteria.md` / `_frozen_criteria__跑后实录.md` | 预登记判据（指纹锚定，不可事后编辑）与跑后实录 |
| `_fuse_qwen.py` | 判据唯一实现，被多个装配脚本 import |
| `_enum_merge.py`、`_assembly_32b.py`、`_validate_semi.py` | 复赛出件关键链路：枚举合并 → 装配 → 校验 |
| `_variant_det_semi.py`、`_predict_gd_3modal.py`、`predict.py` | 推理入口与变体探测 |
| `_build_submission_pkg.py`、`_tidy_root.py`、`_rc.py` | 交付包构建、工作区整理、远端执行通道 |
| `train.py`、`train_competition.py`、`train_lora_gd.py` | 早期训练/微调探索脚本 |
| `config.py`、`prepare_data.py`、`evaluate.py` | 配置、数据准备与评测工具 |
| `requirements.txt` | Python 依赖 |

其余以 `_` 开头的脚本为各阶段的探测、复算与校验工具，均可独立运行。

## 快速开始

```bash
pip install -r requirements.txt
```

推理需要官方数据集与模型权重（本仓库未包含）：

- 数据集：赛事官方初赛/复赛数据集；
- 模型：Qwen3.8-27B（ModelScope `Qwen/Qwen3.8-27B`）本地部署；
- 入口：参考 `技术文档.md` §1.3 的端到端流程，主链路为 `_fuse_qwen.py` → `_enum_merge.py` → `_assembly_32b.py` → `_validate_semi.py`。

## 合规声明

- 仅使用开源模型，本地部署推理，全程无闭源在线 API 调用；
- 仅使用赛事官方测试集，未使用外部数据、未做任何人工标注；
- 所有结果由算法推理产出，后处理为确定性规则，可由管线逐条复算。

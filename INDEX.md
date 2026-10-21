# 2026AiC 目录索引

> 赛题：全球校园人工智能算法精英大赛 · 基于大模型的多模态视觉理解与推理（复赛）
> 本文件说明**每个区的职责**以及**哪些文件动不得**。
> 最近一次整理：2026-09-27（详见 §4）。

---

## 一、目录职责

| 目录 | 职责 | 可否随意改动 |
|---|---|---|
| **`复赛提交材料/`** | **交付包**：README、requirements.txt、`code/`、`result/`。自包含、路径经 `AIC_ROOT` 解析，可独立运行与独立评审。 | 交付物，改动需重新自检 |
| `复赛数据集-基于大模型的多模态视觉理解与推理/` | 复赛数据（5690 条 / 1295 张 / 三模态）。`queries/queries.json` 是键集来源。 | 只读 |
| `初赛数据集-基于大模型的多模态视觉理解与推理/` | 初赛数据（9555 条）。仅作回归对照用。 | 只读 |
| `submission_zips/semifinal/` | 提交件档案，**按实际得分归子目录**（`0.7598/`、`0.7578/`…），扁平目录只留 `SHA256.txt`。 | 归目录规则不得绕过 |
| `semifinal_remote_archive/` | 远端脚本与产物归档（`root/` 为 `/root` 的镜像）。 | 只增不改 |
| `weights/`、`models/`、`GroundingDINO/` | 第三方模型权重与源码（大件，不入交付包）。 | 只读 |
| `viz*`（7 个） | 各阶段目检图。**决策类图必须按真实坐标绘制**——曾因逐层外扩绘制导致误判。 | 只增不改 |
| `_deprecated/` | 历史遗留 + 归档区（含本次整理，见 §4）。 | 见归档区说明 |
| `__pycache__` 等 | 解释器缓存。 | 可删 |

---

## 二、根级活跃文件（153 个）分组

| 组 | 代表文件 | 说明 |
|---|---|---|
| **复赛规则与装配链路** | `_fuse_qwen.py`（判据唯一实现）、`_variant_det_semi.py`、`_enum_merge.py`、`_assembly_32b.py`、`_validate_semi.py`、`_sub_path.py` | 出件关键路径。`_fuse_qwen.py` 被多个脚本 `import` |
| **预登记与跑后实录** | `_frozen_criteria.md`、`_frozen_criteria__跑后实录.md`、`_ord_layer_levers.py` | 见 §3 的纪律约束 |
| **关服清单** | `_root_manifest.txt`（补丁前）、`_root_manifest2.txt`（补丁后） | 远端文件的哈希台账 |
| **环境实录** | `_env_record.txt` | 驱动 / CUDA / 全量 pip freeze / 模型分片 SHA256 |
| **32B 产物** | `_qwen32b_direct.json`（非序数，受保护）、`_qwen32b_direct_ord.json`、`_qwen32b_som_scoped.json` | 哈希被写入跑后实录 |
| **规则层输入** | `_ordinal_cores_r2.json`、`_ordinal_cores_unres.json`、`_enum_raw.json`、`_enum_targets.json`、`_semi_qwen3vl_*.json` | 缺一则离线规则层跑不动 |
| **整理与打包工具** | `_tidy_root.py`（本次整理）、`_build_submission_pkg.py`（交付包构建）、`_rc.py`（远端通道） | 判据可审计、可重跑 |
| 其余 | 各类 `_*.py`/`_*.sh` | 被上述链路引用或属复算/校验工具 |

> 完整清单：`Get-ChildItem -File`。

---

## 三、三条不得违反的纪律

1. **预登记件不可事后编辑**。`_frozen_criteria.md` 的判据在结果出来之前写定；
   其实测修订写在**独立的** `_frozen_criteria__跑后实录.md` 里，并写明所依据的指纹。
   现指纹：`D4A4BC357FAD0F1EE9AD136F756BFEB26B4CF7E949A95E11125BB8389C9E6AE1`。
   **编辑它 = 摧毁"可证明未被编辑"这一属性。**
2. **提交件按得分为准归档**，且同一件不得出现两份（由 `_sub_path.py` 保证）。
3. **看极端样本不能外推整体**；**引用数字前先核实证据来源本身**。
   （两条均为实测教训，详见技术方案文档。）

---

## 四、关于本次整理（2026-09-27）

- **做了什么**：把根级 **270 个"无任何源码引用"的文件** + 4 个无引用目录（3 个 viz、
  `__pycache__`）移入 `_deprecated/归档_20260927/`，按类型分子目录（`py/ sh/ json/
  log/ doc/ zip/ other/ dirs/`）。
- **判据**（可审计、可复跑 `python -B _tidy_root.py` 预演）：满足任一即判"动不得"——
  ① 文件名出现在任一 `.py/.sh/.bat` 源码文本中；
  ② 是 `.py` 且**模块名**出现在源码中（覆盖 `import _fuse_qwen` 这类只出现模块名的引用）；
  ③ 在豁免集内（预登记件、跑后实录、两份清单、复算脚本、`requirements.txt`）。
- **两条硬约束**：**不删除任何文件**；**不移动任何被引用或被指纹锚定的文件**。
- **如何回退**：
  ```
  python -B _deprecated/归档_20260927/取回.py            # 预演
  python -B _deprecated/归档_20260927/取回.py --apply    # 真取回
  ```
- **注意**：归档区内的脚本**不可直接运行**——其路径常量仍指向项目根。需要运行请先取回。
- **整理后复验**（全部通过，退出码 0）：`_ord_layer_levers.py` 复算、交付包
  `_validate_semi.py` 校验、`_test_assembly.py` 夹具回归；`_frozen_criteria.md` 指纹未变。

---

## 五、当前状态与下一步

- 交付件：`复赛提交材料/result/result_semi_enum_trunc.zip`（线上 **0.7598**，本队最高）。
- 待提交件：`复赛提交材料/result/` 之外，32B 非序数件（改动 467 条，预期 **0.7803**）
  在 `submission_zips/semifinal/未提交/result_semi_n32_ordbase.zip`。
- **下一步只有一件事**：等提交机会交该件，按 `得分 × 5690 − 675` 读非序数桶正确数。
- 到 0.8 的主攻方向：**非序数层**（序数侧同族杠杆上界仅 +0.0052，依据见跑后实录）。
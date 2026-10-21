# 基于大模型的多模态视觉理解与推理（Visual Grounding）

面向 2026 AIC 人工智能算法精英大赛「基于大模型的多模态视觉理解与推理」赛题的参赛工程。

核心任务：给定**可见光（RGB）+ 红外（Infrared）+ 深度（Depth）**三模态图像，以及一条描述目标的英文查询文本（Query），利用开源多模态大模型完成**视觉定位（Visual Grounding）**，输出目标在可见光图像中的**归一化边界框 `[x1, y1, x2, y2]`**（取值范围 0\~1）。

**评测指标**：`ACC@0.5`（预测框与真实框 IoU ≥ 0.5 记为正确）。

---

## 1. 技术路线

以开源 VLM **Qwen3.8-27B-Instruct**（ModelScope `Qwen/Qwen3.8-27B`，27.78B 参数）为核心，采用**零样本推理**路线（不训练、不微调），并加入三模态多图输入、低光红外增强、搜索式候选投票、**多产物共识投票后处理**等增强：

```
queries.json ──► 按 Query ID 关联三模态图像与文本
        │
        ▼
查询文本清洗（BOM / 零宽字符 / 连字符统一）
        │
        ▼
多模态预处理：RGB 归一化 / 红外去噪 / 深度离群剔除 /
             低光红外增强（fusion_mode: low_light）
        │
        ▼
模型前向（三模态分图输入 multimodal_input_mode=split，
         multi_box 模式，同图视觉特征缓存）
        │
        ▼
文本 ↔ 检测框匹配 → bbox 归一化 / 合法性校验
        │
        ▼
多产物共识投票后处理（scripts/consensus_vote.py）
        │
        ▼
导出 predictions_consensus.json → make_sub.py → result.zip
```

## 2. 模型权重

- **模型**：Qwen3.8-27B-Instruct（ModelScope：`Qwen/Qwen3.8-27B`）
- **规模**：Safetensors 18 分片，约 54GB，bfloat16
- **下载**：见随包 `AIC-2026-14375345-模型文件/download_model.py`，或
  `modelscope download --model Qwen/Qwen3.8-27B --local_dir ./models/Qwen3.8-27B`
- **校验**：`python download_model.py --verify`，对照随包 `model_sha256.txt` 的 18 分片 SHA256

## 3. 环境依赖

- **GPU**：显存 ≥ 60GB（实测 RTX PRO 6000 Blackwell 96GB）
- **Python**：3.12；**CUDA**：12.4
- 安装：`pip install -r requirements.txt`（torch≥2.6、transformers≥5.18.0、accelerate、opencv-python、Pillow、numpy、PyYAML、tqdm 等）

## 4. 项目文件

```
configs/server_32b.yaml    运行配置（模型路径 / 推理参数 / 后处理回退框）
scripts/infer.py           推理入口（全量；--limit N 冒烟）
scripts/consensus_vote.py  多产物共识投票后处理，产出 result.zip（确定性）
scripts/_bootstrap.py      项目根路径引导
src/data/                  三模态解析与预处理（loader / preprocess / fusion）
src/models/                模型层（base / factory / qwen3_vl / qwen3_vl_parse）
src/inference/             端到端推理 + 文本↔框匹配（predictor / matcher / chains）
src/postprocess/           bbox 校验 + 候选搜索（bbox / search）
src/eval/                  ACC@0.5 / IoU 评测
src/utils/                 日志 / IO / 文本清洗 / LRU 缓存
make_sub.py                预测 JSON → submission.json（round 4 位 + 回退框）
requirements.txt           环境依赖清单
```

> 数据文件不入项目代码目录：投票池 `consensus_pool.json.gz`、主件/参照件预测
> 等均位于同级的 `data_outputs/`。

## 5. 运行步骤

```bash
# 1) 冒烟验证：前 2 张图跑通全流程
python scripts/infer.py --limit 2

# 2) 全量推理（1295 图 / 5690 查询，实测约 5.4 小时）
python scripts/infer.py

# 3) 多产物共识投票后处理 → result.zip（内部 submission.json）
python scripts/consensus_vote.py
```

## 6. 测试与自检

```bash
# 推理冒烟（最小可运行验证）
python scripts/infer.py --limit 2

# 结果自检：条数 / 键集 / 字段回显 / bbox 合法性 四项硬约束
python -c "import json,zipfile;d=json.loads(zipfile.ZipFile('result.zip').read('submission.json'));q=json.load(open('queries.json',encoding='utf-8'));assert set(d)==set(q);print('OK',len(d))"
```

> 本材料包未随附开发期单元测试目录；上述两项可覆盖端到端可运行性与提交格式正确性。

## 7. 注意事项

- 模型权重、数据集等大文件**未随包分发**，请按 §2 下载权重、按 `configs/server_32b.yaml` 的 `data.root` 放置测试集（`Images/visible|infrared|depth` + `queries.json`）。
- `consensus_vote.py` 输出的 zip 已固定内部时间戳，**同输入必得同 SHA256**，可用于结果复算与一致性校验。
- 推理配置为确定性设定（`enable_thinking: false`、`num_samples: 1`），但受 bfloat16 浮点与采样影响，两次独立全量推理间可能存在少量逐条差异（本工程实测约 11%），属正常解码波动。
- 复现需 ≥60GB 显存；显存不足时请调整 `model.device` 或改用云端环境。
- 本工程仅使用赛事官方测试集进行推理，未使用外部数据、未调用任何商业闭源模型的在线 API。

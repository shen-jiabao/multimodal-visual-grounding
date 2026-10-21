# 基于大模型的多模态视觉理解与推理 — LocateAnything 全量融合方案

> **最终提交方案（2026-09-16 锁定）**
> 全球校园人工智能算法精英大赛 · 多模态视觉定位（Visual Grounding）

本项目最终提交使用 **LocateAnything-3B 全量推理 + 与 Grounding DINO 三模态锁版择优融合**。本 README 记录该方案（下文称"方案 L"）的完整方法、复现步骤与结果验证。

- 提交包：`result_locate_merge.zip`（内部 `result_locate_merge.json`，9555 条）
- JSON SHA256：`947B0A0F048E1FC16DDCFC2E59C10FE33807256DF21A203CA8E6ACBC1A418ADD`
- ZIP  SHA256：`D2B4C34B2C9629C34BD293FE508BD40BF126AB62EE929AF9D21EC5D924D8A6AF`

> 原 GD 三模态方案（README.md 方案 A）仍保留为对照基线，已归档；本方案与之择优融合后成为最终版。

---

## 一、任务与数据

- **输入**：可见光 RGB + 红外 + 深度 + 英文文本查询（query）
- **输出**：归一化边界框 `[x1,y1,x2,y2]` ∈ `[0,1]`，需满足 `x1<x2, y1<y2`
- **评估**：`ACC@0.5 = (IoU ≥ 0.5) / 总数`
- **数据**：2000 张可见光图（1903 png + 97 jpg）+ 9555 条查询
- **合规**：仅用开源模型（LocateAnything-3B / GroundingDINO），未调用闭源在线 API

## 二、方案动机

Grounding DINO 三模态早期融合(方案 A)已跑通全量，但在两类问句上存在系统性缺陷：

1. **序数/空间问句**（"second stone pier"、"closest sedan"）：开放词汇检测器收敛到单一高置信框，无法区分同类别内的空间方位/远近。
2. **区域/结构问句**（"metal railing separating path"）：基线框覆盖不完整或语义偏差。

LocateAnything-3B（NVIDIA，Parallel Box Decoding）在指代定位（referring grounding）上能力强于通用检测器。本方案用其在全部 9555 条上做**逐问句单框推理**，再与基线**择优融合**，保证最终结果**不劣于原锁定提交**。

## 三、方法总览（四阶段）

```
test.json / queries.json
        │
        ▼
① 环境装配（隔离 venv + 模型下载 + 数据上传）
        │
        ▼
② 探针验证（4 图 38 问句 → 命中 92.1%，单条 3.83s）
        │
        ▼
③ 全量推理（9555 条，LocateAnything-3B slow/sdpa，50 条/checkpoint，
  0.27 q/s，约 10h）→ locate_bbox_all.json（命中 95.4%）
        │
        ▼
④ 择优融合（vs 基线 result_gd_3modal_fullseq.json）
   → result_locate_merge.json + zip（锁定）
```

## 四、阶段①：环境装配

### 4.1 模型（ModelScope，国内源）

```python
from modelscope import snapshot_download
snapshot_download('nv-community/LocateAnything-3B',
                  local_dir='/srv/models/LocateAnything-3B')
# ~7.7GB：两个 safetensors 分片 + modeling_*.py（trust_remote_code）
```

### 4.2 依赖（隔离 venv，transformers 4.57.1）

```bash
python3 -m venv --system-site-packages /srv/venv-locate   # 用 myconda (py3.13) 建的 python
/srv/venv-locate/bin/pip install --index-url https://pypi.org/simple/ \
  transformers==4.57.1 tokenizers>=0.20 accelerate>=1.5 peft>=0.12 \
  'numpy<2' Pillow opencv-python-headless decord lmdb
```

> 关键：清华镜像对 py3.13 无 transformers wheel，必须走 pypi.org 官方源。
> LocateAnything 官方远程代码依赖 transformers 4.57.1，4.52.4 亦曾验证可用。

### 4.3 推理关键设置（与探针一致）

- `config._attn_implementation = "sdpa"`（text/vision 均设，绕开 magi 自注意力）
- `AutoTokenizer(trust_remote_code=True, fix_mistral_regex=True)`
- 生成 `generation_mode="slow"`、`do_sample=False`、`max_new_tokens=256`
- 图像 resize 到 max 边 2500（LANCZOS）
- bbox 解析：`<box><x1><y1><x2><y2></box>`，0-1000 → 0-1 归一化

## 五、阶段③：全量推理脚本体（`_locate_full.py`）

- 直接以完整 query 作为 prompt，单次生成取**首个 bbox** 为目标
- 每 50 条落盘一次 checkpoint（`/srv/output/locate_bbox_all.json`），支持断点续跑
- 单条 try/except：推理异常不中断批次，标记该条无框，由融合 R0 兜底
- 启动器 `_run_locate_full.sh` 前置校验：数据 2000 图（find \*.png+\*.jpg）、queries/test.json 存在、模型分片完整、GPU 无占用、已有 checkpoint 续跑

## 六、阶段④：择优融合规则（`_merge_locate_gd.py`）

逐条决策，优先级从高到低（**最终版**）：

| 规则 | 条件 | 结果 | 数量 |
| --- | --- | --- | --- |
| **GUARD** | qid ∈ {000282_004/008, 002004_001/002} | 保留基线（硬约束） | 4 |
| **RS** | query 命中连续结构词（railing/fence/barrier/guardrail/handrail 等，含"behind the fence"类句式） | 保留基线 | 380 |
| **R3** | query 命中序数词（first/second/.../last） | 保留基线（基线序数层已验证） | 1955 |
| **R0** | Locate 无框 | 保留基线 | 386 |
| **R1** | Locate 面积 >0.35 且非区域类 | 保留基线 | 94 |
| **R2** | IoU<0.2 且 Locate 面积 >1.5×基线，且非区域/近景/序数空间类 | 保留基线 | 182 |
| **默认** | 其余（含 R2 豁免→保持 Locate 的 KB 类 1659） | 采用 Locate | 6554 |

> 回退合计 = 4+380+1955+386+94+182 = **3001**（与锁定一致）；采用 Locate = 6554。二者相加 9555 ✓。

### 6.1 关键设计（来自探针与人工复查的经验）

1. **缩小 ≠ 漂移**：Locate 的框更精更细（面积常显著小于基线），这是其修正方向。只有**放大**（>1.5×基线）才判为漂移回退。
2. **序数回退**：全量采用`query→单框`直接推理，未走"枚举全候选+取第N"流程，导致 first/second/.../last 未命中正确实例；基线序数层已用该流程人工验证过 → 一律保留基线。`leftmost/rightmost` 等**极值空间词不归入序数**（如 001038_003 最左轿车 Locate 更优）。
3. **连续结构回退**：railing/fence 等长条结构，Locate 单框常只覆盖一小段（实测 012161_004 金属护栏），保留基线整段框。
4. **区域/近景豁免**：lake/grass/ground 等区域类与 closest/nearest 近景类，Locate 大框是正确语义，豁免 R1/R2。

### 6.2 护栏样本保护

4 个人工确认的关键样本任何规则不得改动，融合前后逐条断言一致。

## 七、结果验证

| 指标 | 值 |
| --- | --- |
| 总条目 | 9555（键集 == test.json）|
| 字段完整性 | visible/infrared/depth/query/bbox × 9555，异常 0 |
| bbox 合法性 | 全部满足 `0≤x1<x2≤1, 0≤y1<y2≤1` |
| 采用 Locate | 6554（68.6%）|
| 回退基线 | 3001（序数 1955 + 结构 380 + 无框 386 + R1 94 + R2 182 + 护栏 4）|
| 大框(>0.35) | 基线 230 → 融合 170 |
| 护栏 4 样本 | 零改动 |
| JSON == ZIP 内 | 逐字节一致 |

- 自动校验脚本：`_final_sha256.py`（SHA256 + 键集 + 字段 + bbox 双重校验）
- 人工复核：14 张 v3 抽检图 `viz_out/locate_merge_v3_structure/`（红=融合/Locate，绿=基线），全部接受

## 八、复现步骤（从零）

```bash
# 1. 装配（§4）
# 2. 探针（可选但推荐）
   上传探针包 probe/ + _probe_locale.py → venv python 运行
# 3. 全量推理
   bash /root/_run_locate_full.sh   # 内置全部前置校验，输出 /srv/output/locate_bbox_all.json
# 4. 拉回 + 融合
   scp /srv/output/locate_bbox_all.json .
   python _merge_locate_gd.py        # → result_locate_merge.json + zip
# 5. 校验
   python _final_sha256.py           # → 独立 SHA256 + 键集/字段/bbox
```

## 九、残留风险（已知的未知）

| 风险 | 说明 | 处置 |
| --- | --- | --- |
| ERR 62 KeyError | 62 条查询的 visible 路径推理时 KeyError；文件 md5 远端=本地、路径存在、id 按 224 步长规律出现，根因未定位 | R0 兜底回退基线；影响可控，备案 |
| 阈值泛化 | R1/R2 阈值在 38 条探针校准，泛化到 9555 未全覆盖人工复核 | 抽样 + 抽检图确认；长尾未逐条过目 |
| 多框首框 | 311 条 Locate 多框、融合取首框，清单 `_merge_multibox_notes.json` | 未逐一确认首框最佳 |

## 十、归档与回滚

- 旧 GD 基线：`_deprecated/result_gd_3modal_fullseq.prev20260916.zip`
- 本方案锁定后不可再改；如需回退，从 `_deprecated/` 取回基线并重跑融合

---

**结论**：方案 L 以 LocateAnything-3B 全量推理为骨干、与 GD 基线择优融合，序数/空间/结构问句缺陷通过规则修复，护栏样本受保护，最终结果经自动 + 人工双重验证后锁定。
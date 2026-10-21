# 无人工干预复现说明（两级）

> 目标产物：`submission_zips/0.7798/result_qwen_det_full_qfb_wide.zip`
> 内部 JSON SHA256：`32D42246FE4C003CFAAE6F12C043D73A0CE09C22D2D5F993F75100853C797E63`
> ZIP 容器 SHA256：`AF685528C5E84E5C85E366FC2A709CE321A8192E4F5FDD683675D2623700F47A`
> 线上得分：ACC@0.5 = **0.7798**
> 文档日期：2026-09-19

---

## 复现对象：zip 里的**数据内容**

提交件 zip 内**只有一个文件**，它就是复现对象：

| 项 | 值 |
|---|---|
| 容器内文件 | `result_qwen_det_full_qfb_wide.json`（2.19 MB，**唯一一个**）|
| 内容结构 | JSON：键 = qid，值 = `{visible, infrared, depth, query, bbox}` |
| 条数 | 9555 |
| **数据内容指纹（JSON SHA256）** | `32D42246FE4C003CFAAE6F12C043D73A0CE09C22D2D5F993F75100853C797E63` |
| 自洽校验（与容器无关）| 条数 9555 ｜ 键集 == `test.json` ｜ bbox 全部有序且 ∈[0,1] ｜ 无 `error` 条目 |

> **ZIP 只是外壳。** 本文所称"复现数据内容"一律指**上面这个 JSON 的字节**，其指纹即上表的 JSON SHA256。
> 容器（ZIP 文件本身）的字节可能因 deflate 实现而跨环境不同——那是外壳问题，与数据内容无关（详见 §9 第 1 条）。

## 导读：两级复现，各自对"数据内容"意味着什么

| 级别 | 起点 | 覆盖 | **数据内容（JSON）**能否与提交件一致 | 章节 |
|---|---|---|---|---|
| **A｜从归档中间产物复现** | 4 份已归档的上游产物 | 融合后处理（阶段 7）| ✅ **逐字节一致**（JSON SHA 相同，已实测）| 正文 §0–§9 |
| **B｜整链复现（不依赖任何归档中间产物）** | 赛题数据集 + 模型权重 | 阶段 1–7，三段模型自己跑 | ⚠️ **结构与量级一致、逐条接近，但不保证逐字节相同**（GPU 浮点非确定性）| **附录 B** |

若目标是"**验证这份数据确实是这条规则链算出来的**"，走 **A**（1 条命令、数秒，得到同一个 JSON SHA）。
若目标是"**只从数据集与模型权重出发，把这份数据重新生成一遍**"，走 **B**（需 GPU 与模型权重，先读 §B0）。

---

## 0. 结论先行

本提交件是**纯后处理**产物：**不调用任何模型、不使用 GPU、不需要网络、无随机性、无任何需要人判断的环节**。
给定 **5 个输入文件 + 1 个脚本**，执行 **1 条命令**即可**逐字节复现其数据内容**（即 zip 内那个 JSON）；正常完成时间为**数秒**。

> 该结论已用**净室验证**：把 §3 的输入复制到项目之外的临时目录（`%TEMP%\qfbwide_repro`）重跑，
> 产出的 JSON 与 ZIP **与官方归档件逐字节相同**（见 §8）。净室目录内没有任何项目残留状态。
> 若只关心数据内容：**JSON SHA256 相同即成立**（容器是否相同不影响这一结论）。

---

## 1. 复现边界（必须先说清楚，避免把两类东西混为一谈）

| 层级 | 内容 | 数据内容能否逐字节复现 | 本文覆盖 |
|---|---|---|---|
| **A｜融合后处理（阶段 7）** | 由 4 份既有产物按规则重选 bbox，产出提交件 | ✅ **完全可复现**（纯确定性规则运算）| **是**（正文）|
| **B｜上游构筑链（阶段 1–6）** | 三段模型推理（GD 三模态 / LocateAnything / Qwen3-VL）及其后的本地后处理 | 推理段 ❌ **不保证**（bfloat16 + 算子/调度非确定）；**本地后处理段 ✅ 已实测可逐字节重建** | **是**（附录 B）|

因此本文件覆盖两条完整路径，准确表述分别是：

> **A**：从已归档的上游产物出发，无人工干预复现提交件的**数据内容**（JSON 逐字节一致）。
> **B**：从赛题数据集与模型权重出发重建整链，产出**结构一致、逐条接近**的数据内容（不保证逐字节相同）。

上游 3 份产物已按 SHA256 归档留证（见 §3），想要**与前一次完全相同的数值**时按内容指纹直接复用，不重跑推理。
（上游全量推理实际耗时约 8.08 h、需单卡 17.5 GB 显存；重跑的路径与命令见附录 B，但其产出**不会与归档件逐字节相同**。）

---

## 2. 环境要求

| 项目 | 要求 | 本次实测环境 |
|---|---|---|
| Python | 建议 3.10+（脚本用 `X \| None` 风格注解；**本期只在 3.12.0 上实测通过**，未逐一验证更早版本）| 3.12.0（`C:\Program Files\Python312\python.exe`）|
| Pillow | **必需**（脚本顶层 `from PIL import Image, ImageDraw`，即使不开 `--viz` 也会导入）| 12.2.0 |
| 标准库 | `argparse` / `hashlib` / `json` / `re` / `zipfile` / `pathlib` | 随 Python |

**不需要**：GPU、CUDA、模型权重、网络、数据集图片、任何第三方 API Key。
（`--viz` 选项才会读图，本流程未启用。）

安装依赖（如缺失）：

```powershell
python -m pip install pillow
```

---

## 3. 输入清单（6 个文件，逐一核对 SHA256）

| # | 文件 | 条目数 | SHA256（内容指纹）|
|---|---|---|---|
| 1 | `result_locate_merge_som.json`（基线）| 9555 | `20F1616F7785C77C4129E729767B2CA568BDB254944251692886873597898E97` |
| 2 | `qwen3vl_all.json`（大模型直接定位）| 9555 | `00B612FE3DDAA4610F61DD325F3534114292A6CB99DFFA2D1F80B7886A7214D1` |
| 3 | `qwen3vl_som_all.json`（序数 SoM 记录）| 2014 | `4DB64481596A625E988600B80197964E1BF9944EB0536EB56CF800E24BF42C70` |
| 4 | `_ordinal_cores_v3.json`（序数问句核心词）| 2014 | `531E059DFDDAB233F5C415655B141653A1FC37CAC1844F13387F6C216446A108` |
| 5 | `test.json`（赛题测试集清单，**仅用于键集硬校验**）| 9555 | `DEA1F2D94D8F1A2D235E6A8243D83445AF31AB3E178A0F706CE7ABA23C9A5688` |
| 6 | `_fuse_qwen.py`（融合脚本，595 行）| — | `46A728E6CE1A86CC0FFE85DA150421618FB916D21D441C1875AA4AAD4C3EA15C` |

说明：

- 第 5 项来自赛题数据集目录 `初赛数据集-基于大模型的多模态视觉理解与推理/test.json`。脚本在 `validate()` 中断言
  **输出条数 == 9555 且键集 == test.json 键集**，因此该文件必须提供且版本一致，否则会直接断言失败（有意设计）。
- 第 6 项也是输入：**代码本身即规范**，换一版脚本可能产出不同结果，故必须核对脚本 SHA。

---

## 4. 目录布局

推荐把 6 个文件放在同一个工作目录：

```
<工作目录>\
├─ _fuse_qwen.py
├─ result_locate_merge_som.json
├─ qwen3vl_all.json
├─ qwen3vl_som_all.json
├─ _ordinal_cores_v3.json
└─ test.json
```

若工作目录与原项目不同，**只需修改脚本第 69–70 行的两个常量**（这是唯一的配置项，不含任何判断）：

```python
R = Path(r"d:\Desktop\2026AiC")                                    # 第 69 行：改为 <工作目录>
DATA = R / "初赛数据集-基于大模型的多模态视觉理解与推理"                # 第 70 行：改为存放 test.json 的目录
```

其余 5 个路径常量（`BASE_JSON` / `QD_JSON` / `QS_JSON` / `CORES_JSON` / `TEST_JSON`，第 71–75 行）以及 `VIZ_DIR`（第 76 行）
均**由 `R` / `DATA` 派生**，无需改动。

---

## 5. 复现步骤

### 步骤 1：校验输入（不通过就不要继续）

在工作目录下执行：

```powershell
@'
import hashlib, sys
from pathlib import Path
R = Path(".").resolve()
exp = {
 "result_locate_merge_som.json": "20F1616F7785C77C4129E729767B2CA568BDB254944251692886873597898E97",
 "qwen3vl_all.json":             "00B612FE3DDAA4610F61DD325F3534114292A6CB99DFFA2D1F80B7886A7214D1",
 "qwen3vl_som_all.json":         "4DB64481596A625E988600B80197964E1BF9944EB0536EB56CF800E24BF42C70",
 "_ordinal_cores_v3.json":       "531E059DFDDAB233F5C415655B141653A1FC37CAC1844F13387F6C216446A108",
 "test.json":                    "DEA1F2D94D8F1A2D235E6A8243D83445AF31AB3E178A0F706CE7ABA23C9A5688",
 "_fuse_qwen.py":                "46A728E6CE1A86CC0FFE85DA150421618FB916D21D441C1875AA4AAD4C3EA15C",
}
bad = []
for name, want in exp.items():
    p = R / name
    got = hashlib.sha256(p.read_bytes()).hexdigest().upper() if p.exists() else "(缺失)"
    print(("OK  " if got == want else "FAIL"), name, got[:16])
    if got != want: bad.append(name)
print("输入校验", "全部通过" if not bad else f"失败：{bad}")
sys.exit(1 if bad else 0)
'@ | python -
```

### 步骤 2：执行融合（唯一一条命令）

```powershell
python _fuse_qwen.py --policies det_full_qfb_wide
```

> **必须显式指定 `--policies det_full_qfb_wide`。**
> 脚本默认策略是 `qwen`，不带该参数会产出**另一组文件**（`result_qwen_qwen.*`），那不是本提交件。

### 步骤 3：校验输出

```powershell
@'
import hashlib, zipfile
from pathlib import Path
R = Path(".").resolve()
J = R / "result_qwen_det_full_qfb_wide.json"
Z = R / "result_qwen_det_full_qfb_wide.zip"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
j = sha(J)
inner = zipfile.ZipFile(Z).read("result_qwen_det_full_qfb_wide.json")
i = hashlib.sha256(inner).hexdigest().upper()
print("JSON          ", j)
print("ZIP 内部 JSON ", i, "| 与磁盘一致：", j == i)
print("ZIP 容器      ", sha(Z))
print("JSON == 官方  ", j == "32D42246FE4C003CFAAE6F12C043D73A0CE09C22D2D5F993F75100853C797E63")
print("ZIP  == 官方  ", sha(Z) == "AF685528C5E84E5C85E366FC2A709CE321A8192E4F5FDD683675D2623700F47A")
'@ | python -
```

---

## 6. 预期输出（本次实测原文）

```
基线 9555 | Qwen直接 9555 | SoM 2014 | 序数 2014

== 策略 det_full_qfb_wide ==
  FROZEN           6
  ORD_DET          1279
  ORD_DET_INVALID  2
  ORD_QFB          359
  ORD_QFB_AGREE    83
  ORD_QFB_UNDER    37
  ORD_QFB_UNTRUSTED 25
  ORD_UNDET        706
  QD_AGREE         5526
  QD_DIFF          1784
  QD_INVALID_KEEP  227
  SAME_AS_BASE     1
  SOM              222
  SOM_FAIL_KEEP    7
  改动合计          3788
  其中缩小 1922 / 放大 1866
  宽合并框(>0.35) 169 → 126
  JSON SHA256 32D42246FE4C003CFAAE6F12C043D73A0CE09C22D2D5F993F75100853C797E63
  ZIP  SHA256 AF685528C5E84E5C85E366FC2A709CE321A8192E4F5FDD683675D2623700F47A
```

**读法提示**（这些数字是自洽的，别被"看起来对不上"误导）：

- `ORD_QFB 359` 是**决策数**，其中 1 条算出的框与基线逐位相同（计入 `SAME_AS_BASE 1`），故实际改动 358 条 ⇒ 改动合计 3788。
- 两处 SHA 必须与 §0 给出的值**完全一致**。

---

## 7. 输出文件（3 个）

| 文件 | 是否提交物 | 说明 |
|---|---|---|
| `result_qwen_det_full_qfb_wide.json` | ✅ 是 | 9555 条，字段 `visible/infrared/depth/query/bbox` |
| `result_qwen_det_full_qfb_wide.zip` | ✅ 是 | 内含上述 JSON（提交用的容器）|
| `_audit_qwen_det_full_qfb_wide.json` | ❌ 否 | 3788 条改动明细（含规则名、新旧框、IoU、面积），仅供人工追溯，不参与评分 |

| 硬约束 | 实测 |
|---|---|
| 条数 | 9555 |
| 键集 | == `test.json` |
| 字段集与 bbox 合法性（`x1<x2`、`y1<y2`、∈[0,1]）| 异常 0 |
| `error` 条目 | 0 |
| 面积 >0.35 的宽合并框 | 169 → **126** |
| 冻结样本零改动 | 6 / 6（护栏 4 + 人工确认 2）|

---

## 8. 复现性证据（四项，均已实测）

1. **同环境连跑两次**：JSON、ZIP、ZIP 内部 JSON **三者全部一致**（脚本运行不写任何带时间戳的内容；ZIP 条目时间戳已固定为 `1980-01-01`）。
2. **净室验证**：把 §3 的全部输入复制到**项目之外**的 `%TEMP%\qfbwide_repro`，把路径常量指向该目录后重跑 —— 产出的 JSON 与 ZIP **与官方归档件逐字节相同**（此项排除了"结果依赖项目内残留文件/状态"的可能）。
3. **内部一致性**：ZIP 内 JSON 的 SHA256 == 磁盘 JSON 的 SHA256。
4. **本文 §5 的两段校验脚本已按原文执行通过**：在净室目录中直接运行（脚本内含 `sys.exit`，输入不符会以退出码 1 中断），
   实测输出为"输入校验 全部通过"、"JSON == 官方 True"、"ZIP == 官方 True"。

---

## 9. 已知限制与注意事项

1. **ZIP 容器字节 ≠ 数据内容（本条只影响前者）**：容器条目时间戳已固定，剩余差异只可能来自 **deflate 实现**。本次实测为 CPython 3.12 + zlib 1.2.13，同环境可复现；换用不同 zlib 实现（如 zlib-ng）或不同 Python 大版本，**容器 SHA 可能变化，而内部 JSON 的 SHA256 不变**。
   → 因此**核对数据内容一律比对 JSON SHA256**；容器 SHA 只用于佐证"提交的是哪一个文件"。

2. **`test.json` 版本必须一致**：它是硬校验的基准（键集 + 条数）。版本不符会直接断言失败，这是有意设计而非缺陷。

3. **本流程不验证"定位是否正确"**：ACC@0.5 只能由官方评测返回，数据集不含标注框，本地无法自建评测。
   复现证明的是**产出可逐字节重建**，不是**质量**。质量证据见交付说明 §5（线上八版对照）。

4. **上游产物不可逐字节重建**（见 §1 Tier B）。若确实要重跑上游推理，需 GPU 单卡 ≥17.5 GB 显存、约 8 小时，且**结果不保证逐位一致**；重跑后本流程输出的 SHA 可能随之改变。

5. **不要使用默认策略**：省略 `--policies` 会得到 `qwen` 策略的产物，与提交件不同。

---

## 附：一句话命令速查

```powershell
# 在含 6 个输入文件的工作目录下（已按 §4 调整好第 69–70 行）
python _fuse_qwen.py --policies det_full_qfb_wide
# 期望：JSON SHA = 32D42246FE4C003CFAAE6F12C043D73A0CE09C22D2D5F993F75100853C797E63
#       ZIP  SHA = AF685528C5E84E5C85E366FC2A709CE321A8192E4F5FDD683675D2623700F47A
```

---
---

# 附录 B：整链复现（从赛题数据集与模型权重出发）

> 目标：**不复用任何已归档的中间产物**，从数据集图片与模型权重重建整条链，最终产出提交件。
> 本文与附录 A 的区别：附录 A 的输入里含 3 份"上游产物"，本附录的输入里**没有**它们。

## B0. 三条硬事实（先接受，否则后面会白跑）

1. **整链的 5 段模型推理不可能逐字节重建。** 全链共 7 个阶段，其中 GD 三模态、LocateAnything-3B、Qwen3-VL-8B 三段是 GPU 推理，使用 `bfloat16` + `device_map="auto"`，浮点归约顺序与算子实现随硬件/驱动/库版本变化。**即使完全相同的代码与权重，产出也可能有少量差异**，并会向下游传播。
   → 因此整链复现的**正确预期是"功能等价"**（同样的规则、同样的量级、同样的 ACC 水平），**不是同一个 SHA**。
   → 想要同一个 SHA，只能走附录 A（复用归档产物）。

2. **链条中含 3 处硬编码的"人工确认清单"**（详见 §B4）。它们不是"跑完再手工改数据"，而是**代码里的常量表**，重跑时会自动继承，脚本 SHA 可核。若要求"连这些判断也不允许存在"，则只能删掉这些常量，产出会随之改变。

3. **两处缺口（本次实测确认）**：
   - `dataset_rgb.tar`（当初上传到远端的数据包）**在本地已不存在**，远端 `/srv/dataset` 需要从 `初赛数据集-基于大模型的多模态视觉理解与推理.zip` 重新打包上传。
   - `_weld_som.py` 需要**项目根目录存在 `result_locate_merge.zip`**（它会把该 zip 归档到 `_deprecated/`），缺这一项会在归档步骤 `FileNotFoundError` 退出（注意：**此时 JSON 已经写出**）。

## B1. 阶段总览

| 阶段 | 脚本（本地 → 远端）| 产出 | 运行位置 | 可逐字节重建 |
|---|---|---|---|---|
| 1 | `_analyze_ordinal3.py` + `_launch_qwen_full.py::cleanup()` | `_ordinal_cores_v3.json`（2014）| 本地 CPU | ✅ 纯正则，无模型 |
| 2 | `_launch_qwen_full.py` → 远端 `_qwen_full_pipeline.py`；`_wait_pull_qwen.py` 拉回 | `qwen3vl_all.json`（9555）+ `qwen3vl_som_all.json`（2014）| 远程 GPU（px2）| ❌ |
| 3 | `_run_locate_full.sh` → 远端 `_locate_full.py`；`_pull_analyze.py` 拉回 | `locate_bbox_all.json`（9555）| 远程 GPU（px2）| ❌ |
| 4 | 远端 `predict_gd_3modal.py`（由 `_make_server_predict.py` 生成）+ 本地 `_pipeline_full.py` | `result_gd_3modal_fullseq.json` | 远程 GPU（px1）+ 本地 CPU | 推理段 ❌ / **本地段 ✅ 已实测** |
| 5 | `_merge_locate_gd.py` | `result_locate_merge.json` | 本地 CPU | ✅ **已实测** |
| 6 | `_weld_som.py`（上游 `_run_som.py`→`_probe_som.py` 产 `qwen3vl_som.json`）| `result_locate_merge_som.json`（基线）| 本地 CPU + 远程 SoM 探针 | ✅ **已实测** |
| 7 | `_fuse_qwen.py` | **提交件** `.json` / `.zip` | 本地 CPU | ✅ **已实测** |

## B2. 各阶段命令与要点

### 阶段 1：序数核心词（本地，无模型）

```powershell
python "d:\Desktop\2026AiC\_analyze_ordinal3.py"      # 产出 _ordinal_cores_v3.json（原始版）
python "d:\Desktop\2026AiC\_launch_qwen_full.py"      # 内含 cleanup()：正则清理后就地覆写，并上传远端
```

- `_analyze_ordinal3.py` 仅 `import json/os/re/collections`，**不 import torch/transformers**，纯启发式正则 → **完全确定性**，且无"需 VLM 兜底"的分支（其日志 `_an3.log` 记录"仍无法抽取 0 (0.0%)"）。
- `_launch_qwen_full.py::cleanup()` 同样纯正则（去首 `and/or`、去首冠词、去尾 `facing/mounted` 等），**它把清理结果写回同一路径**（就地覆写），再 SFTP 上传到 `/srv/output/_ordinal_cores_v3.json`。
- ⚠️ 该启动器**必须**提供 `SSH_HOST` / `SSH_PORT` / `SSH_USER` / `SSH_PW` 四个环境变量，且没有默认值；但它同时承担阶段 2 的启动，因此**阶段 1 与阶段 2 的命令耦合在一起**。

### 阶段 2：Qwen3-VL 全量直接定位 + 序数 SoM（远程 GPU）

```powershell
# 本地：注入 SSH 凭据后启动（脚本内部 setsid nohup 远端脚本）
$env:SSH_HOST="<主机>"; $env:SSH_PORT="<端口>"; $env:SSH_USER="root"; $env:SSH_PW="<口令>"
python "d:\Desktop\2026AiC\_launch_qwen_full.py"
# 本地：等待并拉回两份产物（轮询远端日志的完成锚点）
python "d:\Desktop\2026AiC\_wait_pull_qwen.py"            # 可选 --once / --interval 300
# 若曾中断并有图被漏跑，用修复版启动器续跑
python "d:\Desktop\2026AiC\_restart_qwen_full.py"
```

远端实际执行的脚本是 `_qwen_full_pipeline.py`（上传为 `/root/_qwen_full_pipeline.py`），关键常量：

| 项 | 值 |
|---|---|
| 模型 | `/srv/models/Qwen3-VL-8B-Instruct`（ModelScope `Qwen/Qwen3-VL-8B-Instruct`，4 分片）|
| 输出 | `/srv/output/qwen3vl_all.json`、`/srv/output/qwen3vl_som_all.json` |
| 输入 | `/srv/dataset/queries/queries.json`、`/srv/dataset/Images/visible/*`、`/srv/output/_ordinal_cores_v3.json` |
| 生成参数 | `dtype="auto"`、`device_map="auto"`、`do_sample=False`、`max_new_tokens` 按阶段取值 |
| 断点 | `SAVE_EVERY = 50`（每 50 条落盘 direct）；**每张图处理完同时落盘两份 JSON**；重入按 direct/som **双条件**跳过 |
| 解释器 | `/srv/venv-locate/bin/python`（venv 建于系统 myconda py3.13，`--system-site-packages`）|
| 关键库 | `transformers==4.57.1`、`tokenizers==0.22.0`、`torch 2.8`（复用系统）、`accelerate`、`peft==0.12.0`、`numpy<2`、`Pillow>=11`、`opencv-python-headless`、`decord`、`lmdb` |
| 实测代价 | 单卡约 17.5 GB 显存、全量约 8.08 h |

> 环境搭建脚本链：`_setup_px2.sh` / `_venv_px2.sh` / `_venv_fix.sh` / `_setup2_px2.sh`；自检：`_check_qwen3vl_env.py`；下载模型：`_dl_qwen3vl.sh`。

### 阶段 3：LocateAnything-3B 全量（远程 GPU）

```bash
bash /root/_run_locate_full.sh     # 内部：setsid nohup /srv/venv-locate/bin/python -u /root/_locate_full.py
```

`_locate_full.py` 关键常量：模型 `/srv/models/LocateAnything-3B`；输出 `/srv/output/locate_bbox_all.json`（可用环境变量 `QUERIES_JSON` / `OUT_JSON` 覆盖）；`BATCH_SAVE = 50`，**checkpoint 文件即输出文件本身**；`torch_dtype=torch.bfloat16`、`_attn_implementation="sdpa"`、`generation_mode="slow"`、`fix_mistral_regex=True`；图片 resize 至最长边 2500（LANCZOS）；bbox 按 `<box>` 标签解析后 0–1000 → 0–1。本地拉回：`_pull_analyze.py`。

### 阶段 4：GroundingDINO 三模态基线（远程 px1 + 本地后处理）

```bash
bash /root/_run_full_new.sh    # 远端：/root/miniconda3/envs/myconda/bin/python -u predict_gd_3modal.py --out ...
```
```powershell
python "d:\Desktop\2026AiC\_pipeline_full.py" --no-download   # 本地后处理 → result_gd_3modal_fullseq.json
```

- 远端推理脚本由 `_make_server_predict.py` 把 `predict_gd_3modal.py` 的 Windows 路径替换为 `/root/infer/GD`、`/root/dataset/...`、`/root/GroundingDINObase` 后生成（对应本地副本 `_predict_gd_server.py`）。
- 远端环境为 miniconda `myconda`；权重 `groundingdino_swint_ogc.pth` + 本地化 BERT。
- 断点续跑：输出文件已存在则跳过已完成 qid；每 100 条或每 10 分钟落盘。
- `--no-download` 跳过远端拉取；脚本仅在 `result_gd_3modal_full.json` **不存在**时才走 SFTP，因此本地已有该文件时**全程不联网**。

### 阶段 5：Locate × GD 择优融合（本地，纯规则）

```powershell
python "d:\Desktop\2026AiC\_merge_locate_gd.py"
```
输入：`locate_bbox_all.json`、`result_gd_3modal_fullseq.json`、`queries/queries.json`；
输出：`result_locate_merge.json` / `.zip` / `_merge_multibox_notes.json`。

### 阶段 6：SoM 序数焊接 → 基线（本地）

```powershell
python "d:\Desktop\2026AiC\_weld_som.py"
```
输入：`result_locate_merge.json`、`qwen3vl_som.json`（单图 SoM 探针产物，键为 `"000023"`）、`test.json`；**前置：根目录需有 `result_locate_merge.zip`**。
上游 `qwen3vl_som.json` 的产生：`_run_som.py`（上传 `_probe_som.py` 到远端，`/srv/venv-locate/bin/python -u` 执行）→ `_pull_som.py` 拉回。

### 阶段 7：融合出提交件（本地）

```powershell
python "d:\Desktop\2026AiC\_fuse_qwen.py" --policies det_full_qfb_wide
```

## B3. 路径常量差异（决定"换目录要不要改代码"，已实测）

| 脚本 | 根路径来源 | 换工作目录是否需改代码 |
|---|---|---|
| `_pipeline_full.py` | `Path(__file__).resolve().parent` | **不需要** |
| `_merge_locate_gd.py` | `Path(__file__).resolve().parent` | **不需要** |
| `_weld_som.py` | 硬编码（第 17 行） | 需要改 |
| `_fuse_qwen.py` | 硬编码（第 69 行） | 需要改第 69–70 行 |

## B4. 三处人工确认常量（务必知悉）

| 脚本 | 常量 | 内容 | 性质 |
|---|---|---|---|
| `_weld_som.py` | `APPLY` | `["000023_001", "000023_005"]` | **人工看图确认**"宽合并框应为 SoM 窄框"，新框取自 `qwen3vl_som.json` |
| `_merge_locate_gd.py` | `GUARD_SAMPLES` | `{000282_004, 000282_008, 002004_001, 002004_002}` | 4 个护栏样本，任何策略不得改动 |
| `_pipeline_full.py` | `ACCEPTED` | 9 个 qid（`000023_002/003/004`、`001773_005`、`010033_004`、`012826_001`、`014716_003`、`016333_002`、`016774_001`）| **人工确认的序数选择**；另有 `TRUNC_SET={"010019_003"}`，默认 `WRITE_TRUNC=False` 不焊入 |
| `_fuse_qwen.py` | `FROZEN` | 上述 4 护栏 + 那 2 条 | 防回归冻结，任何策略不得覆盖 |

**严格地说**：全链唯一需要人做判断的地方就是这两批 qid 清单（`APPLY` 2 条 + `ACCEPTED` 9 条）；它们一旦写进代码，重跑即自动继承，**不需要任何人在运行时介入**。若要求"连这两批清单也别有"，则须删掉常量重跑，产出会变。

## B5. 本次实测验证结果（真跑，非推断）

在**项目之外的目录**（`%TEMP%\chain_repro`）重建了本地后处理段，并在原布局下补跑阶段 6/7，逐阶段与项目现有产物比对：

| 阶段 | 产物 | 净室/重跑结果 | 现有产物 | 判定 |
|---|---|---|---|---|
| 4b | `result_gd_3modal_fullseq.json` | `87123D1D42C0C7B1…` | `87123D1D42C0C7B1…` | ✅ 逐字节一致（零代码改动）|
| 5 | `result_locate_merge.json` | `947B0A0F048E1FC1…` | `947B0A0F048E1FC1…` | ✅ 逐字节一致（零代码改动）|
| 6 | `result_locate_merge_som.json` | `20F1616F7785C77C…` | `20F1616F7785C77C…` | ✅ 逐字节一致 |
| 7 | `result_qwen_det_full_qfb_wide.json` / `.zip` | `32D42246…` / `AF685528…` | 同 | ✅ 逐字节一致 |

阶段 6 重跑时脚本打印的 2 处修改（即 §B4 的 `APPLY`）：

```
000023_001: SoM#1   旧 [0.002, 0.765, 0.229, 0.949] → 新 [0.0,   0.817, 0.02,  0.888]
000023_005: SoM#5   旧 [0.171, 0.673, 0.344, 0.947] → 新 [0.187, 0.762, 0.219, 0.83 ]
```

复核了 `_deprecated/` 归档未被破坏：`result_locate_merge_som.prev20260917b.{json,zip}` 仍为 `20F1616F…` / `12932E0D…`，
`result_locate_merge.prev20260917.zip` 仍为 `D2B4C34B…`。

**结论**：阶段 4b → 5 → 6 → 7 这四段（全部为确定性后处理）**可从各自输入逐字节重建**；不可逐字节重建的只有阶段 1–4 中的模型推理段。

## B6. 已知缺口与注意事项（均为实测）

1. **`dataset_rgb.tar` 缺失**：本地只有数据集 zip，远端 `/srv/dataset` 需自行重新打包上传（`_up_dataset.py` 引用的 `SRC` 路径已失效）。
2. **阶段 6 的前置**：`_weld_som.py` 需要根目录有 `result_locate_merge.zip`。本次用 `submission_zips/result_locate_merge.zip`（容器 `D2B4C34B…`，与 `_deprecated/` 版本同）补回，随后重跑成功。缺失时它会在归档步骤报错退出，**但 JSON 已写出**——不要误判为"没跑出结果"。
3. **无顶层编排脚本**：阶段需按序手工调用。`run.bat` / `run.sh` / `run_competition.bat` 是 GD 方案 A 的入口，**不在本链上**。
4. **`_predict_gd_server.py` 的上传器未找到**：仅找到生成器 `_make_server_predict.py` 与通用上传工具 `_remote_run_file.py` / `_remote_upload_run.py`。
5. **`predictions_v2_highres_slow.json`、`predictions_grounding.json` 的生成脚本未在项目内找到**（它们是阶段 4 的输入）。
6. **多数中间 zip 是"带时间戳容器"**：例如 `_weld_som.py` 产出的 `result_locate_merge_som.zip` 本次容器为 `D6CC5481…`（条目时间戳即运行时刻 2026-09-19 03:31），**下一次运行会不同**。故链条中对 zip 的核对同样应比对**其内部 JSON**。
7. **远端依赖自备**：GPU 主机、模型权重（Qwen3-VL-8B-Instruct / LocateAnything-3B / GroundingDINO+BERT）与模型许可；`_launch_qwen_full.py` 的 SSH 凭据仅经环境变量注入，不落盘。

## B7. 两级复现各自能保证什么（汇总）

以**数据内容（zip 内那个 JSON）**为准：

| 复现起点 | 覆盖阶段 | 数据内容能保证什么 |
|---|---|---|
| 4 份归档产物（附录 A）| 7 | ✅ JSON **逐字节一致**（SHA 相同）|
| 阶段 3/4 产物（本地）| 5 → 7 | ✅ 三段后处理逐字节一致（已实测）|
| 阶段 4 之后（远程推理产物 + 本地）| 4b → 7 | ✅ 四段逐字节一致（已实测）|
| 赛题数据集 + 模型权重（附录 B 全链）| 1 → 7 | ⚠️ 硬条件全满足、逐条接近；**不保证逐字节相同** |

## B8. 走 B 路线时，"数据内容复现成功"如何判定

**必须 100% 满足的硬条件**（与附录 A 同一套校验，与容器无关）：

- 条数 == 9555
- 键集 == `test.json`
- 字段集 == `{visible, infrared, depth, query, bbox}`
- 每条 bbox 满足 `0 ≤ x1 < x2 ≤ 1`、`0 ≤ y1 < y2 ≤ 1`
- 无 `error` 条目
- 若保留 §B4 的常量：4 护栏样本零改动、`APPLY` 那 2 条等于 SoM 选中框

**不能期待**：JSON SHA256 与提交件相同（原因见 §B0 第 1 条）。

**建议额外报告**（用于说明"接近到什么程度"）：与提交件逐条比对 bbox 的 IoU 分布（中位数、以及 IoU<0.5 的条数）。

> ⚠️ **诚实声明**：本项目**从未真正从零重跑过全链**（阶段 1–3 的远端环境与 `dataset_rgb.tar` 均已不可得），
> 因此**上面这条"接近程度"没有任何实测数值**，本文件不给预期区间——给了就是编造。
> 若要获得该数值，必须在具备 GPU 的环境里实际重跑整链并逐条比对。

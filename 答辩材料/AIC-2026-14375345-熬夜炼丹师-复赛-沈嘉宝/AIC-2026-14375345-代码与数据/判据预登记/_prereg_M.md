# 预登记 M —— 客户端对齐「对方规格」：三模态分图 + 5 条规则 + multi_box + max_new=1024 + thinking 关闭

> **本文件性质**：判据事前文档。一经写下**不得事后编辑**（项目纪律），只能另立新件。
> 生效时间：2026-10-05。硬件：AutoDL 单卡 **RTX PRO 6000 Blackwell 96GB**，
> 模型：`Qwen/Qwen3.8-27B`（`Qwen3_5ForConditionalGeneration`，27.781 B，全 BF16，未量化）。

---

## §0 一句话目的

把**对方队伍**（同为 Qwen3.8-27B 三模态、5690 条全量跑 5.5 小时得 **0.8092**）的那套
**客户端配置**搬到我们的管线上跑一遍，检验「我们能复现到多少」，并读出它相对我们
当前最高 **0.7963**（I 件 q38）的增量。

---

## §1 ★先声明：本件**不是单变量实验**

> **★修订 1（2026-10-05，开卡后、**尚未产生任何读数**时）**
> 用户指定：三模态**合成一张图**，每次只传这一张（而非分图三张）。
> 故因子 1 的取值由 `=3`（分图）改为 **`=3m`（横向三联合并图，见 §3.5）**。
> 该修订发生在**任何推理与得分产生之前**，不构成"事后挑参数"；
> 此后本文件不得再改，只能另立新件。

本项目此前各件（D / I / K / L / N）都坚持「只改一个因子」。**本件明确放弃该纪律**，
因为目的是**打包复现对方配置**，不是一个可归因的对照实验。以下 6 个因子**同时变化**：

| # | 因子 | 我们现状（I 件 q38 0.7963） | 本件 M |
|---|---|---|---|
| 1 | 模态输入 | `QWEN_MODALITIES=1`（**只读可见光单图**） | **`=3m`：三模态横向三联合并成一张图**（修订 1） |
| 2 | 提示词 | 无附加规则 | **追加 5 条规则**（§3.1 逐字） |
| 3 | 请求粒度 | 每 query 一次前向，`CONC=8` 并发 | **multi_box：同图 ≤20 条合并一次前向** |
| 4 | `max_new_tokens` | 256 | **1024** |
| 5 | thinking | **未显式设置**（模板默认 ⇒ 开，effort 默认 `xhigh`） | **显式关闭**（`enable_thinking=False`） |
| 6 | 作用范围 | 4587 非序数（序数走 Phase B SoM） | **5690 全量直接定位**（含序数 1103） |

⇒ **本件得分一动，不得归因于上述任何单项。** 要拆分因子须另立新件。
⇒ 因子 6 的变更同时改变了**读数口径**，见 §5。

---

## §2 为什么不保持单变量（决策留痕）

对方方案的价值就在「这一整套」：他们的 0.8092 是这套配置跑出来的。
只搬其中一项无法回答「我们能复现吗」这个真实问题。故本件以**对表**为第一目标，
以**归因**为放弃项——这是主动选择，不是疏忽。

---

## §3 规格（逐字冻结）

### §3.1 五条规则（`RULES5_TEXT`，英文逐字如下）

```
Additional rules that MUST be followed:
1. Ordinal questions ("the Nth from the left/right/top/bottom"): count from
   the outermost edge of the image inward, starting at 1.
2. Left/right is always from the viewer's own perspective (the image's
   left/right), never from the object's own perspective.
3. For a small part (a button, knob, handle, switch or label), draw the box
   around that part ONLY - never around the whole object that contains it.
4. Draw a tight box that hugs the object; do not include background.
5. If the description mentions a colour, only match objects whose visible
   colour matches it.
```

**插入位置**：两条既有提示词（`FULL_DIRECT_PROMPT` / `THREE_IMG_PROMPT`）在
`Description: {q}\n` 之后各加一个 `{rules}` 槽。规则关（默认）时该槽展开为空串
⇒ 提示词与改造前**逐字节相同**。

### §3.2 multi_box 协议（`MULTIBOX_PROMPT`，仅当 `MODALITIES=3` 可用）

- 分组：同一张图内**待处理** query 按原顺序切成 ≤`MAXBOX`(=20) 的块，每块**一次前向**。
- 描述以 `1. … / 2. … / …` 编号列出，**各描述相互独立**（同一物体可被描述多次）。
- 输出约定（**新约定，故不复用旧断言**）：

```
Output ONLY a JSON array of exactly {n} objects, one per description, in the same
order as above, like
[{"idx": 1, "bbox_2d": [x1, y1, x2, y2], "label": "object"}]
where coordinates are integers normalized to 0-1000
(x1,y1 = top-left, x2,y2 = bottom-right). Use {"idx": i, "bbox_2d": null}
for a description that has no matching object.
```

- 解析（`parse_multibox`）：① 先取首个 `[` 到末个 `]` 整体 `json.loads`，按 `idx`
  定位（`idx` 缺失或越界时退回数组下标）；② 失败则正则逐段取 `"idx": k` 与
  `"bbox_2d": [...]`。
- 坐标缩放**复用 `parse_boxes`**（把单个框包成 `[{"bbox_2d":[…]}]` 再解析），
  保证「坐标空间适配只实现一处」。
- 某条解析不到 ⇒ 该条 `bbox=None`（后续装配器按既有覆盖率规则保留基线）。

### §3.3 thinking 关闭

OpenAI 请求体加 `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`。
依据：`chat_template.jinja` 第 46 行 `if enable_thinking is undefined or enable_thinking is true`
⇒ 不传即视为开，且 `reasoning_effort` 默认 `xhigh`；第 165–166 行是唯一关闭分支。
**未知项**：本机 vLLM 0.23.0 的 OpenAI server 是否接受该字段——须在起服后用 1 条请求实测，
若不接受则改为**起服参数** `--default-chat-template-kwargs '{"enable_thinking": false}'`
（属基础设施路径替换，不动提示词/解码/并发）。

### §3.4 其余

- `max_new_tokens=1024`（Phase A / multi_box 用；Phase B 的 512/16 不变，本件不跑 Phase B）。
- 红外/深度的预处理沿用预登记 D §3（`load_ir` / `load_depth`，`DEPTH_MAX=20000`，无效深度置黑）。

### §3.5 三模态合并图（`QWEN_MODALITIES=3m`，修订 1）

**版式**（`merge3`）：**横向三联，等宽等高**，左→右 = 可见光 / 红外 / 深度。
以可见光那张的尺寸为准，红外与深度各自缩放到同尺寸（数据集内三模态本就同尺寸，
实测 .png 组均 1920×1080、.jpg 组均 640×360 ⇒ 通常是恒等变换）。
格间留 **4 px 浅灰分隔线**（便于人眼在 G3 证据图上分辨边界）；
**不写任何文字标注**（避免文字被当成待定位目标）。

**坐标口径（关键）**：提示词要求模型输出**以左侧可见光那一格自身为 0–1000**
（`MERGED_PROMPT` 的输入描述段写明）。因此 `parse_boxes` 的 ÷1000 与既有约定**完全一致，
无需任何换算**。

**兜底开关**：若实测发现模型实际给的是**整张合并图**的坐标（表现：框全挤在左侧 1/3），
置 `QWEN_MERGE_SPACE=whole` ⇒ `remap_merged` 把 x 乘 3 换回可见光格（三格等宽）。
判据：G3 证据图（20 条）里框是否落在可见光格内的目标上。

---

## §4 判据（**事先写定**）

### §4.1 硬闸门（不过就停）

| 闸门 | 判据 | 失败动作 |
|---|---|---|
| **G0** | vLLM registry 登记 `Qwen3_5ForConditionalGeneration` | 已过（2026-10-05 02:47 实测，架构总数 365） |
| **G1** | 权重 18 分片 sha256 与 HF 官方值双验通过 | 已过（18/18，0 异常） |
| **G2** | 20 条冒烟：**可解析率 ≥16/20** | <16 ⇒ 格式不兼容，**停**，先修解析 |
| **G3** | 20 条冒烟：人眼方向正确 **≥12/20** | 8~11 ⇒ 放行但登记"接地能力弱"；<8 ⇒ **停** |
| **G4** | 全量跑完后**有框率 ≥95%** | <95% ⇒ 输出格式与解析不匹配，**作废本次读数**，先修 |

### §4.2 主判据（对表，事后不得改）

设本件得分为 `S_M`，既有实测：我们最高 `0.7963`（I 件 q38）、对方 `0.8092`。

| 判定 | 区间 | 含义 |
|---|---|---|
| **复现成功** | `S_M ≥ 0.8092 − 0.01` = **≥0.7992** | 对方方案可复现 |
| **部分复现** | `0.7963 ≤ S_M < 0.7992` | 至少超过我们现有最好 |
| **无增益** | `0.791 ≤ S_M < 0.7963` | 与 q38 同水平或略低 |
| **负** | `S_M < 0.791` | 明显负（低于我们 L 件的负向水平），**封口** |

### §4.3 观察量（不设阈值）

- 改动条数、净增减（相对基线交付件）。
- multi_box 每次前向的平均 query 数、平均耗时（用于估算全量 ETA）。
- `raw` 里是否出现 `<think>`（thinking 是否真的关掉了）。

---

## §5 读数口径（**本件特有，与 I 件不同**）

- I 件口径：`得分 × 5690 − 689 = 非序数桶正确数`（因 I 件序数走 Phase B 原样保留）。
- **本件是「5690 全量直接预测」⇒ 不再做桶扣除**：`S_M` 本身即可与 `0.7963` / `0.8092`
  直接比较。若误用 I 件口径会系统性低估。

---

## §6 发车清单

1. 用户面板**切有卡模式**（`/dev/nvidia*` 出现）。
2. `_get_serve_q38.sh` ⇒ TP=1（单卡 96 GB ≥ 75 GB），起 vLLM，逐 attention 后端重试。
3. `_m_thinking_probe.py`：1 条请求（同图同 query）各打一次，判定 thinking 是否关闭、
   以及 vLLM 是否接受 `chat_template_kwargs`（§3.3）。
4. `_q38_gate.sh`（在 M 环境下运行，自动走 3m + 规则 + multi_box）⇒ G2/G3（20 条冒烟）。
   同时人工看 `q38_gate_vis/_sheet.png` 判 §3.5 的坐标口径（框是否落在可见光格内的目标上）。
5. 过闸 ⇒ **`_launch_M.sh`**（已注入 `QWEN_MODALITIES=3m QWEN_RULES5=1
   QWEN_MULTIBOX=1 QWEN_MAXNEW=1024 QWEN_THINKING=0 QWEN_ONLY_NONORD=0`，
   内部 `exec` 复用 `_launch_q38.sh` 的闸门）⇒ 全量 5690。
6. 装配 ⇒ 提交件 ⇒ 读数（§5）。
7. 记录 ETA；multi_box 首图跑完后按实测速率决定是否继续。

---

## §7 回退与封口

- 硬闸门任一失败 ⇒ 停，先修闸门，不产出提交件。
- `S_M < 0.791` ⇒ 按 §4.2 判负，**封口**（不再在本方向上试参数）。
- 本件**不删任何既有文件**；若 M 负，既有 0.7963 交付件不受影响（平台按历史最高记录）。

---

## §8 已知风险（事前登记）

| # | 风险 | 若发生怎么办 |
|---|---|---|
| R1 | vLLM 不接受 `chat_template_kwargs` 字段 | 改起服参数 `--default-chat-template-kwargs`（§3.3） |
| R2 | multi_box 输出格式不合约定（尤其 20 个对象的顺序/嵌套） | G4 兜住；必要时降 `MAXBOX` 到 5 或 10 重跑（属调参，须另立新件记录） |
| R3 | `enable_thinking=False` 后输出质量下降（规则 1–5 需要推理） | 观察量记录；不中途改判据 |
| R4 | 20 条合并后单次前向序列变长，OOM | 降 `MAXBOX` |
| R5 | 全量 5690 在 multi_box 下耗时与对方 5.5h 差异大 | 记录 ETA，按实测决定 |

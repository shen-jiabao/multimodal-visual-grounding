# 自证文档：交付件 `result_semi_q38_nonord` 来源溯源

> 参赛团队编号 **AIC-2026-14375345** ｜ 团队名称 **熬夜炼丹师** ｜ 队长 沈嘉宝
> 赛题：全球校园人工智能算法精英大赛 · 算法挑战赛 · **基于大模型的多模态视觉理解与推理**
>
> **本文用途**：就交付件 `result_semi_q38_nonord.zip`（线上 **ACC@0.5 = 0.7963**）的**产出服务器**做一次**可独立复核的来源溯源**，
> 证明该文件由本队在自有云端实例上、以本队代码推理产出，并**如实登记取证中发现的记载不一致**。
>
> **取证时间**：2026-10-15（Asia/Shanghai）｜**取证方式**：SSH 只读采集（不修改、不删除远端任何文件）
> **取证服务器**：`connect.westd.seetacloud.com:43504`（AutoDL 容器 `autodl-container-a19d43b48e-bedc3b24`）

---

## 〇、一页速览

| 项 | 值 |
| --- | --- |
| 交付件 | `submission_zips/semifinal/0.7963/result_semi_q38_nonord.zip`（内部同名 JSON，5690 条） |
| 交付件 SHA256 | `FB6C2E471A4F5E8907318FF540A5CAABAEC17A29ECAFC17EB651F512A4AE8F35`（**本次独立复算一致**） |
| 线上成绩 | 复赛 **ACC@0.5 = 0.7963**（本队排行榜最高分） |
| 产出机制 | 跨代换模型：Qwen3.8-27B 重跑非序数层 4587 条，单变量替换 **519** 处 ⇒ 净 **+101** 条 |
| 产出服务器 | `connect.westd.seetacloud.com:43504`（容器 `autodl-container-a19d43b48e-bedc3b24`） |
| 溯源结论 | **成立**——7 项现场证据与归档记载逐项吻合，另有 1 项环境记载**不一致**（见 §六，已如实登记） |
| 成品位置 | 交付件 zip **不在服务器上**；服务器只保留推理过程中的间产物与日志（成品已搬回本地装配打包） |

---

## 一、自证主张与核验手段

| # | 主张 | 本文提供的核验手段 | 章节 |
| --- | --- | --- | --- |
| **A** | 交付件**由本队自有实例产出** | 服务器上留存同一模型的权重、起服日志、原始推理产物与补跑清单 | §三 / §四 |
| **B** | 交付件的**改动面可本地复算** | 本地对拍 0.7963 与 0.7786 两件，逐条复算改动数 | §五 |
| **C** | 取证结论**不含未经核实的断言** | 明确区分"现场实测""归档转引""未能证实"三类，并登记 1 项记载冲突 | §六 |

---

## 二、取证对象

| 项 | 实测值 |
| --- | --- |
| 连接方式 | SSH，`connect.westd.seetacloud.com:43504`，用户 `root` |
| 容器主机名 | `autodl-container-a19d43b48e-bedc3b24` |
| 容器内网 IP | `172.17.0.9` |
| 容器镜像 | Ubuntu 22.04.5 LTS |
| 实例数据盘创建时间 | 2026-10-05 02:06:20 +0800 |
| 工作目录 | `/root/autodl-tmp/rivermind-data`（软链 `/root/rivermind-data`） |
| CPU / 内存 | 208 核 / 1007 GB |
| 磁盘占用 | 数据盘 `autodl-tmp` 78 GB；系统盘 overlay 30 GB（仅用 212 MB） |
| 数据盘内容 | `models/` 52 G｜`uvcache/` 9.1 G｜`venv_q38/` 9.0 G｜`dataset/` 7.5 G｜`cache/` 366 M｜`tmp/` 203 M｜`q38_gate_vis/` 54 M｜`output/` 3.4 M |

---

## 三、证据链：7 项现场实测与归档记载逐项吻合

比对基准：《`submission_zips/semifinal/SHA256.txt`》中 0.7963 件条目（第 307–358 行）。

| # | 归档记载 | 服务器现场实测（证据文件） | 判定 |
| --- | --- | --- | --- |
| 1 | 模型 `Qwen3.8-27B`；vLLM **0.23.0** | `vllm38.log`：`model = /root/rivermind-data/models/Qwen3.8-27B`；`Initializing a V1 LLM engine (v0.23.0)` | **一致** |
| 2 | G0 闸门：vLLM registry 登记 `Qwen3_5ForConditionalGeneration` | `vllm38.log`：`Resolved architecture: Qwen3_5ForConditionalGeneration`；`_q38_serve.log`：`架构登记 Qwen3_5ForConditionalGeneration: True` | **一致** |
| 3 | 起服加 `--disable-custom-all-reduce` 退回 NCCL | `vllm38.log` `non-default args`：`'disable_custom_all_reduce': True` | **一致** |
| 4 | `--limit-mm-per-prompt {"image":1,"video":0}` | `vllm38.log`：`'limit_mm_per_prompt': {'image': 1, 'video': 0}` | **一致** |
| 5 | torch **2.11.0+cu130** | `venv_q38/bin/python` 实测 `2.11.0+cu130`；`_q38_serve.log` 同 | **一致** |
| 6 | 装配桶报告「**1 无框 → 保留基线 329**」 | `output/m_direct.json` 实测 **无框数 = 329** | **一致（精确）** |
| 7 | （归档未记，现场补充） | `_m_retry_qids.json` 为列表，**长度 = 329**，且与第 6 项的 329 个 qid **逐个相同、顺序相同** | **闭环** |

**补充独立核验（交付件指纹）**

| 核验项 | 结果 |
| --- | --- |
| 交付件 zip SHA256 复算 | `FB6C2E471A4F5E8907318FF540A5CAABAEC17A29ECAFC17EB651F512A4AE8F35` ⇒ **与索引一致** |
| zip 内条目名 | `['result_semi_q38_nonord.json']` ⇒ **与 zip 基名相同** |
| zip 内 JSON 条数 | 5690 |
| zip 内 JSON ⇄ 磁盘同名 json | 键集相同、**逐条差异 0**（仅序列化缩进/换行不同，故字节级 SHA256 不同） |

---

## 四、因果链重建（服务器现场时间线）

服务器上与该交付件相关的文件时间戳全部落在 **2026-10-05**，可完整复现生产顺序：

```text
2026-10-05 02:11   _q38_env.sh             环境检查脚本
2026-10-05 02:50   _q38_build.sh           建 venv_q38（vLLM 0.23.0 / torch 2.11.0+cu130）
2026-10-05 02:53   _q38_dl.sh              下载 18 分片权重（断点续传 + 逐片 sha256 双验）
2026-10-05 03:15   q38_env.log / _q38_build.out
2026-10-05 03:51   q38_dl.log              全部 18 个分片就位
2026-10-05 04:16   _q38_gate.py / _get_serve_q38.sh
2026-10-05 04:31   _q38_gate20.json / _q38_gate20_panel.json / q38_gate_vis/*.png   ← G2/G3 闸门产物
2026-10-05 04:36   _launch_q38.sh
2026-10-05 08:20   output/m_direct.json     27B 全量直出：5690 条，其中 329 条模型未给出框
2026-10-05 11:20   vllm38.log              vLLM 重启（补跑会话）
2026-10-05 11:23   _m_retry_qids.json      把 329 条无框条目列为补跑清单
2026-10-05 11:39   output/m_retry.json     补跑结果
2026-10-05 11:40   output/m_direct_filled.json   5690 条，无框 = 0（补跑回填完成）
2026-10-05 13:42   _qwen_full_pipeline.py  管线最后修改
2026-10-05 13:43   _make_submission.py     装配脚本落盘（服务器上**从未执行**，无 zip 产出）
—— 此后成品搬回本地，由本地装配器合成 519 处改动并打包上传 ⇒ 线上 0.7963 ——
```

**关键闭环**：`m_direct.json` 的 329 条无框条目 ⇒ `_m_retry_qids.json` 的 329 条补跑清单 ⇒ 补跑后 `m_direct_filled.json` 无框归零。
这条链同时解释了归档中"无框 329 → 保留基线"这一桶的**数据来源**，不是事后书写的数字，而是**服务器上可查的现场事实**。

**原始推理产物样例**（`output/m_direct.json`，键 `000001_001`）：

```json
{"bbox": [0.633, 0.72, 0.672, 0.822], "n": 1,
 "raw": "```json\n[\n\t{\"idx\": 1, \"bbox_2d\": [211, 720, 224, 822], \"label\": \"A squatting person\"}, ..."}
```

---

## 五、本地可复算部分（主张 B）

用本地两件成品 json 直接对拍（无需 GPU）：

| 项 | 值 |
| --- | --- |
| 新件 `0.7963/result_semi_q38_nonord.json` | 5690 条 |
| 底稿 `0.7786/result_semi_n32bf16_nonord.json` | 5690 条 |
| ★ 改动（IoU < 0.5） | **519** ⇒ 与归档记载**完全一致** |
| 与基线一致（IoU ≥ 0.5） | 5065 |
| 面积 > 0.35 | 106 |
| 无框 | 0 |

**线上读数反推**：

```text
0.7963 × 5690 = 4530.947 → 4531
0.7786 × 5690 = 4430.234 → 4430
Δ = (0.7963 − 0.7786) × 5690 = 100.7 ≈ +101 条
净成功率 = 100.7 / 519 = 19.4%
```

**一处可解释的口径差**：本地粗算得「面积 > 0.35 = 106」，归档桶报告为「100」。
原因是我在**回填之后**比对，那 329 条"模型无框"的条目已等于基线框，被我计入了"与基线一致"（5065 = 3639 + 329 − 6）；
即差异来自**桶判定优先级**，非数据不符，两侧非序数条目总数同为 4587。

---

## 六、如实登记：一处环境记载不一致（未证实项）

**这是本次取证中发现的唯一一项冲突，不作掩饰，原文并列如下。**

| 来源 | 记载的推理硬件环境 |
| --- | --- |
| 归档《`SHA256.txt`》0.7963 件条目（§338–346） | **2×A100-SXM4-40GB** / NV12 NVLink / P2P=True / 驱动 580.126.09；**TP=2**；KV cache **7.61 GiB/卡**；**186,075 tokens** |
| 服务器现场 `_q38_serve.log`（2026-10-05 11:22）与 `vllm38.log` | **1×NVIDIA RTX PRO 6000 Blackwell Server Edition（97887 MiB ≈ 95 GB）**；**TP=1**；Available KV cache **32.2 GiB**；**458,752 tokens**；`Model loading took 51.08 GiB` |

**两侧各自自洽，但指向不同硬件。** 最可能的解释（**属推断，非实测**）：

- 服务器上的 `vllm38.log` 在 11:20 被**覆盖重写**，现存日志只覆盖 **11:20 起始的补跑会话**；
- 而**全量跑**（产出 `m_direct.json`，08:20 落盘）的那次起服会话日志**已不存在**；
- 因此归档记载的可能正是**全量跑那次会话**的环境，而现存日志是**补跑会话**的环境。

**本队未能从现场唯一还原全量跑的硬件环境**，故本文：
1. **不主张**归档记载的 A100 环境为真；
2. **不主张**现场 RTX PRO 6000 即全量跑的硬件；
3. 明确标记该项为 **"记载冲突·原因待定"**，请评审以本条为准，勿直接引用归档中的环境表述。

**另一项未能证实**：容器内 `nvidia-smi` 返回 `Permission denied`，本次**无法直接从容器内读取 GPU 型号**。
现存 GPU 型号信息**全部来自本队运行期自采日志**（`_q38_serve.log` / `vllm38.log`），非平台出具。

---

## 七、取证命令（评审/复核者可照做）

以下为本次实际执行的**只读**命令（等价形式）：

```bash
# 0) 连接
ssh -p 43504 root@connect.westd.seetacloud.com

# 1) 主机与环境标识
hostname; hostname -I; cat /etc/os-release | head -2; nproc; free -g | head -2
df -h

# 2) 交付件相关脚本/日志是否存在（应**无任何 zip**，仅有 _make_submission.py）
find /root /autodl-tmp -type f \( -iname '*result*' -o -iname '*submission*' \) \
  -not -path '*/venv_q38/*' -not -path '*/uvcache/*' -not -path '*/models/*'

# 3) 起服参数与架构解析（证据 1–5）
grep -n "non-default args\|Resolved architecture\|vLLM engine (v0" \
  /root/rivermind-data/vllm38.log
grep -n "架构登记\|torch 2.11.0" /root/rivermind-data/_q38_serve.log
/root/rivermind-data/venv_q38/bin/python -c "import torch;print(torch.__version__)"

# 4) ★无框 329 与补跑清单闭环（证据 6–7）
/root/rivermind-data/venv_q38/bin/python - <<'PY'
import json
R='/root/autodl-tmp/rivermind-data'
d=json.load(open(R+'/output/m_direct.json',encoding='utf-8'))
nobox=[k for k,v in d.items() if not isinstance(v.get('bbox'),list) or len(v.get('bbox') or [])!=4]
q=json.load(open(R+'/_m_retry_qids.json',encoding='utf-8'))
print('m_direct 条数=',len(d),'无框=',len(nobox))
print('retry 清单长度=',len(q))
print('两者是否逐个相同且同序 =', nobox==q)
PY

# 5) output 间产物指纹
md5sum /root/autodl-tmp/rivermind-data/output/*.json
```

**本次实测输出（关键三项）**：

```text
m_direct 条数= 5690 无框= 329
retry 清单长度= 329
两者是否逐个相同且同序 = True
```

**本地侧复核命令**：

```powershell
# 交付件指纹（期望 FB6C2E47…8F35）
certutil -hashfile "d:\Desktop\2026AiC\submission_zips\semifinal\0.7963\result_semi_q38_nonord.zip" SHA256
```

---

## 八、结论与边界

### 成立的部分

1. **产出服务器已定位**：交付件由 `connect.westd.seetacloud.com:43504`（容器 `autodl-container-a19d43b48e-bedc3b24`）产出。
   模型权重、起服日志、原始推理产物、补跑清单**四类物证齐备**，与归档记载 **7 项逐项吻合**。
2. **"无框 329"不是事后书写的数字**：它与 `_m_retry_qids.json` 构成现场可查的**闭环**（329 条无框 → 329 条补跑 → 无框归零）。
3. **交付件指纹独立复算一致**：`FB6C2E47…8F35`；zip 内条目名与基名一致；zip 内 JSON 与磁盘 json 逐条零差异。
4. **改动面 519 本地复算一致**，与归档记载完全相符。

### 边界（如实陈述）

1. **成品不在服务器上**：全盘无任何 `result_semi*` / `submission` / `predictions` 文件，无与任务相关的 zip；
   服务器上仅有 `_make_submission.py`（未执行）。**装配与打包发生在本地**，本文不覆盖该段的证据。
2. **线上分数 0.7963 是唯一外部事实**：无法本地复算，`Δ = +101` 与 `19.4%` 均由其派生；
   若平台读数有误，派生量会整链偏移。
3. **一处环境记载冲突（§六）**：归档记为 2×A100-SXM4-40GB，现场日志记为 1×RTX PRO 6000 Blackwell。
   **本队不作单边主张**，标注为"记载冲突·原因待定"。
4. **GPU 型号无法从容器内直读**（`nvidia-smi` 权限拒绝），现存型号信息全部来自运行期自采日志。

### 一句话结论

**该交付件的"生产现场"在这台服务器上完整可查（模型、日志、原始产物、补跑闭环四证齐备），
但成品本身与最终装配发生在本地；同时归档中的一处环境记载与现场日志冲突，已如实登记，不作掩饰。**

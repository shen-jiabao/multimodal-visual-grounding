# 复赛提交材料说明

| 项 | 内容 |
| --- | --- |
| 参赛团队编号 | AIC-2026-14375345 |
| 团队名称 | 熬夜炼丹师 |
| 队长 | 沈嘉宝 |
| 队员 | 李宇鑫、祁皓然 |
| 赛题 | 基于大模型的多模态视觉理解与推理 |

---

## 一、材料清单

```
AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝/          根文件夹
├── AIC-2026-14375345-技术方案.PDF                技术方案（12 页，含核心代码）
├── README.md                                     本说明文件
├── AIC-2026-14375345-模型文件/                    模型权重获取与校验
│   ├── download_model.py                         权重下载脚本
│   ├── model_sha256.txt                          18 分片 SHA256
│   └── 模型文件说明.txt                           模型基本信息
├── AIC-2026-14375345-代码与数据/                  代码与数据
│   ├── code27b/                                  推理工程（预处理 / 推理 / 后处理）
│   ├── data_outputs/                             管线输入与运行记录
│   ├── 版本生成脚本/                              各版本件的生成脚本
│   ├── 代码与数据说明.txt                         目录与产物来源说明
│   └── 投票池来源表.txt                           投票池 32 份成员来源
└── AIC-2026-14375345-赛题结果文档/                结果文件
    ├── result.zip                                提交结果（内部 submission.json）
    └── SHA256.txt                                哈希与生成说明
```

---

## 二、项目简介

本方案面向「基于大模型的多模态视觉理解与推理」赛题：给定同一场景的
**可见光（visible）、红外（infrared）、深度（depth）** 三模态图像与一条
描述目标的英文查询，输出目标在可见光图像中的**归一化边界框**
`[x1, y1, x2, y2]`（取值 0~1）。

技术路线为**预训练多模态大语言模型零样本推理 + 多产物共识投票后处理**：

1. 以 Qwen3.8-27B-Instruct 为基座，三模态图像分图输入视觉编码器；
2. 以 `multi_box` 模式一次生成多个候选框，经文本与检测框匹配、
   坐标归一化后得到每查询一个 bbox；
3. 对主件与参照件的**真分歧位**（两框互 IoU < 0.5）施加**多产物共识投票**，
   仅当「主件零票、参照件满票」时采纳参照件框。

不训练、不微调，未使用任何外部训练数据。

**关于训练环节**：本方案直接使用 Qwen3.8-27B-Instruct 的**官方基线权重**，
未做任何训练、微调或超参数调优（零样本推理），因此不含训练环节代码。
推理所需的全部处理逻辑（数据预处理 / 输入构造 / 解码 / 后处理）均在
`code27b/` 内，可直接运行。

---

## 三、大文件与独立链接

本作品文档及其链接**不包含模型权重、数据、环境等大文件**：随包材料
全部为代码、配置与说明，合计约 7 MB。大文件按下方**独立链接**获取：

| 大文件 | 体积 | 获取方式（独立链接） |
| --- | --- | --- |
| 模型权重 Qwen3.8-27B-Instruct | 约 54 GB | ModelScope 官方地址 `Qwen/Qwen3.8-27B`；另附 `download_model.py` 与 18 分片 SHA256 校验 |
| 赛题测试数据 | — | 由赛事方提供，未随包分发 |
| 运行环境 | — | 依赖清单见 `code27b/requirements.txt`（未提供镜像） |

模型基本信息：

| 项 | 内容 |
| --- | --- |
| 模型 | Qwen3.8-27B-Instruct |
| 参数量 | 27.78B |
| 精度 | bfloat16 |
| 体积 | Safetensors 18 分片，约 54 GB |

下载与校验：

```bash
pip install modelscope
python download_model.py                 # 下载到 ./models/Qwen3.8-27B
python download_model.py --verify        # 逐分片比对 SHA256
```

分片哈希清单见 `AIC-2026-14375345-模型文件/model_sha256.txt`。

---

## 四、运行配置

| 项 | 要求 |
| --- | --- |
| GPU | 显存 ≥ 60 GB（实测 RTX PRO 6000 Blackwell 96 GB） |
| 系统 | Linux（Ubuntu），磁盘 ≥ 100 GB |
| Python | 3.12 |
| CUDA | 12.4 |

关键依赖：torch ≥ 2.6、transformers ≥ 5.18.0、accelerate、opencv-python、
Pillow、numpy、PyYAML、tqdm（完整清单见 `code27b/requirements.txt`）。

---

## 五、项目文件介绍

```
code27b/
├── configs/server_32b.yaml          运行配置
├── scripts/
│   ├── infer.py                     推理入口（全量 / --limit 冒烟）
│   ├── consensus_vote.py            共识投票后处理管线
│   ├── pool_dimensions.py           投票池的配置规则维度说明
│   └── _bootstrap.py                路径注入
├── src/
│   ├── config.py                    配置加载
│   ├── data/                        数据加载、三模态预处理与融合
│   ├── models/                      Qwen3-VL 封装与输出解析
│   ├── inference/                   推理流程、文本匹配、约束解析
│   ├── postprocess/                 bbox 归一化、候选搜索
│   ├── eval/                        ACC@0.5 与 IoU 计算
│   └── utils/                       IO、文本清洗、缓存、日志
├── make_sub.py                      预测 JSON → submission.json
├── requirements.txt                 环境依赖
└── README.md                        工程说明
```

数据与产物（`data_outputs/`）：

| 文件 | 用途 |
| --- | --- |
| `queries.json` | 赛题测试集索引（官方提供，未修改） |
| `predictions_27b.json` | 主件：27B 全量推理输出（5690 条） |
| `predictions_ref.json` | 参照件：另一次全量推理输出 |
| `consensus_pool.json.gz` | 投票池：32 套产物坐标 |
| `env_record.txt` | 实测环境快照 |

各产物的来源与复现流程见 `代码与数据说明.txt`；
投票池 32 个成员的来源见 `投票池来源表.txt`。

---

## 六、测试命令

```bash
# 1. 环境准备
pip install -r code27b/requirements.txt

# 2. 权重下载（见第三节）
python download_model.py

# 3. 放置测试集：data.root 下 Images/visible|infrared|depth + queries.json

# 4. 冒烟测试（少量样本，验证环境与链路）
python code27b/scripts/infer.py --config code27b/configs/server_32b.yaml --limit 20

# 5. 全量推理（约 5.4 小时 / 单卡 96GB）
python code27b/scripts/infer.py --config code27b/configs/server_32b.yaml

# 6. 共识投票后处理，产出 result.zip
python code27b/scripts/consensus_vote.py

# 7. 校验提交件（条数 / 键集 / 字段回显 / bbox 合法性）
#    校验逻辑见 code27b/make_sub.py（坐标规范化与合法性检查）
```

**可复现性**：后处理为确定性规则，zip 时间戳已固定，
**同输入必得同 SHA256**。

---

## 七、注意事项

1. **推理结果与提交件一一对应**：`AIC-2026-14375345-赛题结果文档/result.zip`
   为最终提交件，内部文件名为 `submission.json`。
2. **结果文件唯一**：全部材料中仅此一个结果 zip；其余 `.json` 均为管线输入，
   不是备选结果。
3. **字段原样回显**：`visible` / `infrared` / `depth` / `query` 四个字段
   与 `queries.json` 完全一致，未做任何修改；仅 `bbox` 为预测值。
4. **坐标精度**：bbox 为归一化坐标，保留 4 位小数。
5. **异常回退**：无法解析的查询回退到占位框 `[0.49, 0.49, 0.51, 0.51]`。
6. **运行环境**：需 ≥60 GB 显存的 GPU；CPU 环境无法完成全量推理。
7. **路径说明**：`configs/server_32b.yaml` 中的路径为推理时所用路径，
   复现时请按实际环境调整 `data.root` 与 `model.name_or_path`。

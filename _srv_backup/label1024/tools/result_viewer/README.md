# 结果查看器（result_viewer）

将云端推理结果（`predictions.json` 的 bbox）叠加到数据集三模态图上，供人工逐条质检浏览。后端为 Python 标准库（`http.server`），前端为原生 HTML/JS/Canvas 单页，**零第三方依赖、离线可用**（无 CDN）。

## 快速启动

```powershell
E:\AIC人工智能算法精英大赛-基于大模型的多模态视觉理解与推理赛道\.venv\Scripts\python.exe tools\result_viewer\server.py [端口]
```

- 端口可选，默认 `8000`，例如 `server.py 8123`。
- 浏览器打开 `http://127.0.0.1:8000`。
- 启动日志会打印识别到的 run 列表及 predictions/submission 可用性。

## 功能

| 功能 | 说明 |
|---|---|
| 结果叠加 | bbox 叠加到 visible / infrared / depth 三模态图，可切换主图模态 |
| 多 run 同屏对比 | 顶栏「对比 run」多选（最多 4 个），不同颜色叠加各 run bbox，附图例与坐标列表 |
| query 中文翻译 | 右侧栏显示 query 原文 + 译文（离线词典，逐字符匹配） |
| 人工标注 | 对 / 错 / 存疑 + 备注，导出 JSON 到 `outputs/复赛/audit/` |
| 结果源切换 | `predictions.json`（默认）与 `submission.json` 可选 |
| 搜索 | 支持样本 ID 精确匹配与 query 关键字子串过滤，回车跳转 |
| 快捷键 | ← / → 翻页；1 / 2 / 3 切模态；m 快速标记「存疑」 |

## 界面布局

- **顶栏**：结果源、主 run、对比 run、搜索框、模态切换按钮。
- **主画布**：当前模态图 + 各 run bbox 叠加，右上角图例。
- **右侧栏**：样本 ID、三模态缩略图（点击切换主图）、各 run 归一化坐标、人工标注区。
- **底栏**：进度（第 x / 5690 条）、快捷键提示、上一张 / 下一张。

## 目录结构

```
tools/result_viewer/
├── server.py              # 后端服务（标准库，~180 行）
├── static/
│   ├── index.html         # 前端单页
│   └── query_zh.json      # 中文翻译词典（5234 条唯一 query）
└── _work/                 # 词典源材料（非运行必需）
    ├── unique_queries.txt # 5234 行唯一 query 原文（行号 = 索引）
    ├── zh_part_01~09.json # 翻译分片
    └── merge_check.py     # 分片与原文对账脚本
```

## 数据格式（已核实）

- 样本键：`000001_001`（图ID_序号），共 5690 条。
- `predictions.json`：`{样本键: [x1, y1, x2, y2]}`，归一化 0~1，无置信度。
- `submission.json`：`{样本键: {visible, infrared, depth, query, bbox}}`，路径格式 `Images/visible/000001.png`。
- `dataset/queries/queries.json`：`{样本键: {visible, infrared, depth, query}}`。
- 图片：`dataset/Images/{visible,infrared,depth}/000001.png`（1920×1080，约 3.2MB/张）。

## API 路由

| 路由 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 前端页面 |
| `/api/runs` | GET | run 列表及预测源可用性 |
| `/api/meta?run=X&q=关键词` | GET | 样本键列表；`q` 可选，按 query 子串过滤 |
| `/api/sample?run=X&id=Y&compare=Z1,Z2&src=predictions` | GET | 单样本详情（query、译文、三模态路径、各 run bbox） |
| `/img/<modal>/<图ID>.png` | GET | 图片（modal 限 visible/infrared/depth） |
| `/api/annotations` | POST | 保存人工标注 |

## 人工标注与导出

标注项：`ok`（对）/ `wrong`（错）/ `doubt`（存疑），可附备注；同一项再点取消。标注仅存于前端内存，需手动导出：

```json
{
  "run": "run_20260930_012029_qwen_id1024",
  "src": "predictions",
  "generated_at": "2026-09-30T16:19:55",
  "items": {
    "000001_001": { "verdict": "ok", "note": "定位准确" },
    "000463_002": { "verdict": "doubt", "note": "" }
  }
}
```

导出文件：`outputs/复赛/audit/manual_review_<run>_<时间戳>.json`。

## 翻译词典维护

`static/query_zh.json` 由 `_work/` 源材料合并生成，key 与原文**逐字符一致**（原文含首尾空格/双空格）。修改某条译文：

1. 在 `_work/unique_queries.txt` 定位原文行号；
2. 修改对应 `zh_part_*.json` 分片；
3. 运行 `_work/merge_check.py` 对账（输出缺失/多余条数）；
4. 合并写回 `static/query_zh.json`（与初次合并相同的合并逻辑）。

## 安全与限制

- 图片路由、run 名、样本键均做严格校验，路径穿越请求返回 404（已验证）。
- 标注仅内存态，刷新页面即丢失，请及时导出。
- 同一张图承载多条 query，浏览顺序即样本键在 `predictions.json` 中的顺序。
- 词典为人工翻译，个别译文可能有误，以原文为准。

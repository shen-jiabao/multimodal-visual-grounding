# 标注工具（子项目）

独立人工标注工具：为测试集图片标注真值 bbox（评价依据），支持人机对比、提交结果比对、多人协作认领制（已实现）。

## 目录结构

```
annotation_tool/
├── annotate.py              # 主程序：标注服务 / score 比对 / gen-machine 生成机器标注
├── collab.py                # 多人协作核心：会话分配 / 认领 / 心跳 / 释放合并 / 超时清理 / 退出保护
├── annotate_ui.html         # 标注界面·桌面端（前端），由 annotate.py 从同目录加载
├── annotate_ui_mobile.html  # 标注界面·移动端（前端，手机/窄屏自动分发，?view=mobile|desktop 可强制切换）
├── roster.html              # 认领名单页（前端），5 秒刷新，管理员可强制释放
├── translations.json        # 查询描述离线中译表（8829 条唯一英文描述 → 中文）
├── test_collab.py           # 多人协作回归测试（双会话全流程，独立 out-dir + 端口运行）
├── .gitignore               # 忽略运行产物（__pycache__ / .lock / .tmp / outputs/ 标注数据）
└── docs/
    ├── 标注工具使用说明.md   # 完整使用说明（运行、标注流程、导入、评分、参数、多人协作）
    └── 多人协作标注策划.md  # 多人合作标注功能策划（方案定稿，已实现）
```

## 与根项目的关系

| 依赖 | 位置（根项目） | 说明 |
|---|---|---|
| 数据集 | `dataset/`（自动定位，无需参数） | 三模态图像与查询定义，**保留在根项目读取**，不复制进子项目 |
| 评测脚本 | `scripts/evaluate.py` | 根项目推理评测流程的一部分（与 `src/eval/metrics.py` 配套），消费子项目导出的 `answer.json` |
| 机器标注来源 | 根项目 `outputs/` 下 `predictions*.json` / `submission*.json` | 推理 / 提交流程产物，导入与比对时自动扫描读取 |

子项目只包含标注工具自身文件；数据集与评测脚本等根项目资源跨目录引用，互不复制。

## 快速开始

```bash
# 任意目录运行均可（数据集与输出目录自动定位）
.venv\Scripts\python.exe scripts\初赛\annotation_tool\annotate.py
```

启动后自动打开浏览器（http://127.0.0.1:8765），详细用法见 `docs/标注工具使用说明.md`。

## 运行生成的文件（均在子项目 `scripts/初赛/annotation_tool/outputs/`）

```
outputs/
├── annotations/          # 标注数据工作区
│   ├── annotations.json          # 人工标注主文件（交还 / 超时 / 强制释放 / 退出时合并写入）
│   ├── tmp_<会话名>.json         # 各会话在手图片的临时标注（合并后删除）
│   ├── sessions.json             # 会话表（会话名 / 是否管理员 / 创建与最后活跃时间）
│   ├── assignment.json           # 认领表（图片 → 会话 / 认领时间 / 心跳时间）
│   ├── release_log.json          # 释放 / 合并日志（保留最近 200 条）
│   ├── answer.json / answer_full.json  # 答案文件（点击"导出答案"生成，合并视图）
│   ├── imported_annotations.json # 机器标注（导入的模型预测框，与人工标注独立保存）
│   └── score_report.json         # 提交结果比对分数报告
└── backup/               # 人工标注备份（annotations_backup_YYYYMMDD/）
```

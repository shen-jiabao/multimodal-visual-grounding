# -*- coding: utf-8 -*-
"""生成本地临时文件 _predict_gd_server.py：把 predict_gd_3modal.py 的
Windows 路径常量替换为服务器路径，便于上传执行。"""
from pathlib import Path

SRC = Path(r"d:\Desktop\2026AiC\predict_gd_3modal.py")
DST = Path(r"d:\Desktop\2026AiC\_predict_gd_server.py")

text = SRC.read_text(encoding="utf-8")

REPL = [
    # GD 源码目录
    ('GD = Path(r"d:\\Desktop\\2026AiC\\weights\\gdino_ms\\AI-ModelScope\\GroundingDINO")',
     'GD = Path(r"/root/infer/GD")'),
    # 数据根目录
    ('DATA_ROOT = Path(r"d:\\Desktop\\2026AiC\\初赛数据集-基于大模型的多模态视觉理解与推理")',
     'DATA_ROOT = Path(r"/root/dataset/初赛数据集-基于大模型的多模态视觉理解与推理")'),
    # 模型配置与权重
    ('CFG_PATH = r"d:\\Desktop\\2026AiC\\weights\\gdino_ms\\AI-ModelScope\\GroundingDINO\\groundingdino\\config\\GroundingDINO_SwinT_OGC.py"',
     'CFG_PATH = r"/root/infer/GD/groundingdino/config/GroundingDINO_SwinT_OGC.py"'),
    ('CKPT_PATH = r"d:\\Desktop\\2026AiC\\weights\\groundingdino_swint_ogc.pth"',
     'CKPT_PATH = r"/root/infer/GD/weights/groundingdino_swint_ogc.pth"'),
    # 测试索引/输出（远端无 test.json，queries/queries.json 结构一致）
    ('DEFAULT_TEST_JSON = DATA_ROOT / "test.json"',
     'DEFAULT_TEST_JSON = DATA_ROOT / "queries" / "queries.json"'),
    ('DEFAULT_OUT_JSON = Path(r"d:\\Desktop\\2026AiC\\result_gd_3modal.json")',
     'DEFAULT_OUT_JSON = Path(r"/root/infer/result_gd_3modal_v2.json")'),
    # BERT 复用远端 GroundingDINObase
    ('os.environ.setdefault("BERT_DIR", str(GD / "bert-base-uncased"))',
     'os.environ.setdefault("BERT_DIR", "/root/GroundingDINObase")'),
]

for old, new in REPL:
    if old not in text:
        raise SystemExit(f"[缺失] 未找到待替换常量:\n{old}")
    text = text.replace(old, new)

DST.write_text(text, encoding="utf-8")
print(f"已生成 {DST}")

"""数据加载模块。

负责解析赛题 `queries.json`，按 Query ID 关联三模态图像路径与查询文本，
并按“图像”分组（同一张图可能对应多条查询），供推理器批量处理。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

from src.config import Config
from src.utils.io import load_json, read_image_rgb, read_image_uint16


@dataclass
class QuerySample:
    """单条查询样本。"""
    query_id: str
    visible_path: str
    infrared_path: str
    depth_path: str
    query: str
    bbox: List[float] | None  # 若数据带标签则回填（赛题测试集无标签，为 None）


@dataclass
class ImageGroup:
    """同一张可见光图像下的所有查询样本分组。"""
    image_key: str            # 例如 "000002"
    visible_path: str
    infrared_path: str
    depth_path: str
    samples: List[QuerySample]


def _abs_join(base: str, rel: str) -> str:
    return os.path.normpath(os.path.join(base, rel))


def load_queries(cfg: Config) -> List[QuerySample]:
    """读取 queries.json，返回全部查询样本列表（保持文件顺序）。"""
    raw = load_json(cfg.queries_path)
    data_root = cfg.data_root
    samples: List[QuerySample] = []
    for qid, item in raw.items():
        samples.append(
            QuerySample(
                query_id=qid,
                visible_path=_abs_join(data_root, item["visible"]),
                infrared_path=_abs_join(data_root, item["infrared"]),
                depth_path=_abs_join(data_root, item["depth"]),
                query=item["query"],
                bbox=item.get("bbox"),
            )
        )
    return samples


def group_by_image(samples: List[QuerySample]) -> Dict[str, ImageGroup]:
    """按可见光图像分组查询样本。

    使用图片相对路径作为 key，保证同图多查询被聚合。
    """
    groups: Dict[str, ImageGroup] = {}
    for s in samples:
        key = s.visible_path
        if key not in groups:
            groups[key] = ImageGroup(
                image_key=os.path.splitext(os.path.basename(key))[0],
                visible_path=s.visible_path,
                infrared_path=s.infrared_path,
                depth_path=s.depth_path,
                samples=[],
            )
        groups[key].samples.append(s)
    return groups


def load_sample(sample: QuerySample) -> dict:
    """双路径加载三模态图像。

    PNG 深度：uint16 单通道毫米值（depth_is_metric=True）。
    JPG 深度：8bit RGB 伪灰度，取 R 通道（depth_is_metric=False）。
    """
    ext = os.path.splitext(sample.visible_path)[1].lower()
    is_jpg = ext in (".jpg", ".jpeg")

    rgb = read_image_rgb(sample.visible_path)          # uint8 (H,W,3)
    try:
        ir = read_image_rgb(sample.infrared_path)      # uint8 (H,W,3)
    except Exception:
        ir = rgb  # 损坏时用RGB兜底

    try:
        if is_jpg:
            depth = read_image_rgb(sample.depth_path)[..., 0]
            depth_is_metric = False
        else:
            depth = read_image_uint16(sample.depth_path)
            depth_is_metric = True
    except Exception:
        depth = None
        depth_is_metric = False

    return {
        "rgb": rgb,
        "infrared": ir,
        "depth": depth,
        "depth_is_metric": depth_is_metric,
        "query_id": sample.query_id,
        "query": sample.query,
        "orig_size": rgb.shape[:2],   # (H, W)
        "is_jpg": is_jpg,
    }

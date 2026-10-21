# -*- coding: utf-8 -*-
"""Grounding DINO 三模态早期融合推理，生成比赛提交结果。

路线（对应 a.txt 二/三节）：
- 将可见光 RGB(3)、红外(1)、深度(1) 拼接为 5 通道输入。
- 修改 SwinT PatchEmbed 第一层卷积输入通道 3->5，复制预训练前 3 通道权重。
- 输出归一化 [x1,y1,x2,y2] bbox，生成 result.json。

用法：
    python predict_gd_3modal.py [--n 条数] [--test_json ...] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as Ftv
from PIL import Image
from torchvision.ops import nms

# ----------------------------- GD 环境 -----------------------------
GD = Path(r"d:\Desktop\2026AiC\weights\gdino_ms\AI-ModelScope\GroundingDINO")
sys.path.insert(0, str(GD))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("BERT_DIR", str(GD / "bert-base-uncased"))

from groundingdino.datasets import transforms as T  # noqa: E402
from groundingdino.models import build_model  # noqa: E402
from groundingdino.util.misc import clean_state_dict  # noqa: E402
from groundingdino.util.slconfig import SLConfig  # noqa: E402

from models.gd_fusion import modify_model_feature_fusion  # noqa: E402
from models.gd_lora import inject_lora, load_lora  # noqa: E402
from models.pose_classifier import (  # noqa: E402
    PoseClassifier,
    load_ir_gray,
    parse_pose_word,
    prepare_patch,
)

# ----------------------------- 路径/配置 -----------------------------
DATA_ROOT = Path(r"d:\Desktop\2026AiC\初赛数据集-基于大模型的多模态视觉理解与推理")
CFG_PATH = r"d:\Desktop\2026AiC\weights\gdino_ms\AI-ModelScope\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py"
CKPT_PATH = r"d:\Desktop\2026AiC\weights\groundingdino_swint_ogc.pth"
DEFAULT_TEST_JSON = DATA_ROOT / "test.json"
DEFAULT_OUT_JSON = Path(r"d:\Desktop\2026AiC\result_gd_3modal.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
# 收集候选框的置信阈值（低于 BOX_THRESHOLD，保留多个框供空间语义排序）。
# 设为 0.15 而非 0.20：边缘实例（如远左车辆）的 logit 常贴在 0.2 附近，
# 跨进程/并发推理时数值轻微浮动会导致其偶尔掉出候选池，0.15 留出裕量。
CANDIDATE_THRESHOLD = 0.15

# 输入图像参考边（与 load_image 保持一致的 RandomResize 语义）
IMAGE_REF = 800
MAX_RESIZE = 1333
# 深度排序时允许的候选框最大面积（归一化比例），避免"合并整图框"误作最近/最远目标
MAX_BOX_AREA_FRAC = 0.35

# 疑似"纵向堆叠/人+前景物合并"的高瘦框：像素空间高宽比阈值（> 该值视为合并长条框）
TALL_BOX_ASPECT = 2.2
# 目标类别 -> 换 prompt 重检测用的上位词（用于拆解"人 + 前景物"的纵向合并实例）。
# 原理：具体类别词（如 doll）与 person 的特征重叠导致合并；上位词（toy）逐实例
# grounding 更清晰，能让 Grounding DINO 重新拆出每个独立物体。键为上位词，值为
# 命中该上位词的具体名词子串（在 query 小写文本中匹配）。
SUPERNOUN_MAP = [
    ("toy", ["doll", "teddy", "bear", "plush", "stuffed", "toy", "robot",
             "figurine", "lego", "block", "brick", "puppet"]),
    ("animal", ["bird", "dog", "cat", "deer", "horse", "cow", "sheep", "goat",
                "swan", "duck", "goose", "chicken", "rooster", "hen", "rabbit",
                "fox", "wolf", "tiger", "lion", "elephant", "monkey", "fish",
                "butterfly", "insect", "animal", "pig", "peacock", "zebra",
                "camel", "antler", "elk", "sika", "cattle", "panda"]),
    ("vehicle", ["car", "truck", "bus", "van", "suv", "sedan", "coupe", "jeep",
                 "motorcycle", "motorbike", "scooter", "moped", "bike", "bicycle",
                 "train", "vehicle", "cart", "trolley", "carriage", "tractor",
                 "lorry", "tram"]),
    ("furniture", ["chair", "table", "desk", "sofa", "couch", "bed", "cabinet",
                   "shelf", "wardrobe", "stool", "bench", "counter", "drawer"]),
    ("dishware", ["cup", "bottle", "bowl", "plate", "glass", "mug", "jar",
                  "pot", "vase", "bucket", "kettle", "teapot", "basket"]),
    ("electronic", ["phone", "laptop", "computer", "keyboard", "clock", "lamp",
                    "camera", "screen", "monitor", "switch", "panel", "tv",
                    "television"]),
]

# 竖向实体：本体就高瘦，其高瘦框是正确答案，不应被候选扩充或高瘦框过滤误删/误拆。
# 覆盖人形 + 树/柱/塔 + 门/墙/柜等常见竖向目标。
TALL_ENTITY_WORDS = [
    # 人形
    "person", "people", "man", "woman", "child", "boy", "girl",
    "pedestrian", "walker", "worker", "driver", "lady", "gentleman",
    "human", "men", "women",
    # 植物
    "tree", "trunk", "stalk", "bamboo", "palm", "cactus", "plant",
    # 柱/塔/杆
    "pole", "pillar", "column", "post", "tower", "mast", "flagpole",
    "lamppost", "streetlight", "pylon",
    # 建筑
    "building", "house", "skyscraper", "chimney",
    # 门/墙/柜（竖向结构）
    "door", "gate", "wall", "wardrobe", "cabinet", "bookcase", "bookshelf",
    "cupboard", "locker", "refrigerator", "fridge",
]


# ----------------------------- 图像加载 -----------------------------
def _imread_unicode(path: str) -> np.ndarray | None:
    """兼容中文路径图像读取。"""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def _to_single_tensor(gray: np.ndarray, target_size):
    """灰度图直接 resize 到目标尺寸并 ImageNet 归一化，返回 (1, H, W)。"""
    gray_pil = Image.fromarray(gray).resize(
        (target_size[1], target_size[0]), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(gray_pil, dtype=np.float32) / 255.0)
    mean = float(r"0.485")
    std = float(r"0.229")
    t = (t - mean) / std
    return t.unsqueeze(0)                                   # (1, H, W)


def load_multimodal_image(rgb_path, ir_path, depth_path):
    """加载三模态并统一 transform。

    Returns:
        image_source: RGB numpy (h, w, 3)，归一化坐标参考。
        multimodal: (5, H_, W_) 拼接待测张量。
        h, w: 原图高宽。
        depth_raw: 原分辨率深度图 (h, w) uint16，用于空间语义排序。
    """
    compose = T.Compose([
        T.RandomResize([IMAGE_REF], max_size=MAX_RESIZE),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 1. 可见光 RGB（基准尺寸）
    rgb_bgr = _imread_unicode(rgb_path)
    assert rgb_bgr is not None, f"RGB 读取失败: {rgb_path}"
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    image_source = np.asarray(rgb)
    h, w = image_source.shape[:2]
    rgb_pil = Image.fromarray(image_source)
    rgb_t, _ = compose(rgb_pil, None)        # (3, H_, W_)
    _, H_, W_ = rgb_t.shape

    # 2. 红外（单通道）
    ir = _imread_unicode(ir_path)
    assert ir is not None, f"IR 读取失败: {ir_path}"
    ir_gray = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY) if ir.ndim == 3 else ir
    ir_t = _to_single_tensor(ir_gray, (H_, W_))             # (1, H_, W_)

    # 3. 深度（16bit 单通道）
    raw = np.fromfile(depth_path, dtype=np.uint8)
    depth_full = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    assert depth_full is not None, f"深度图读取失败: {depth_path}"
    depth_full = (cv2.cvtColor(depth_full, cv2.COLOR_BGR2GRAY)
                  if depth_full.ndim == 3 else depth_full)  # 原分辨率
    depth = cv2.resize(depth_full, (W_, H_),
                       interpolation=cv2.INTER_NEAREST)
    depth_t = _normalize_depth(depth)                          # (1, H_, W_)

    # 4. 拼接为 5 通道：RGB(3) + IR(1) + Depth(1) = 5
    multimodal = torch.cat([rgb_t, ir_t, depth_t], dim=0)      # (5, H_, W_)
    return image_source, multimodal, h, w, depth_full


def _normalize_depth(depth: np.ndarray) -> torch.Tensor:
    """深度归一化到 [0,1]，输出 (1, H, W) 张量。
    16-bit 深度单位 mm；0 / <300mm 视为无效区域。
    """
    d = depth.astype(np.float32)
    valid = (d >= 300) & (d <= 19999)
    if valid.any():
        mean_v = d[valid].mean()
        d[~valid] = mean_v
    d = np.clip(d, 300, 19999)
    d = (d - 300.0) / (19999.0 - 300.0)          # -> [0,1]
    t = torch.from_numpy(d).unsqueeze(0)         # (1, H, W)
    return t


# ----------------------------- 模型 -----------------------------
def modify_model_first_conv(model: nn.Module, in_channels: int = 5) -> nn.Module:
    """将 SwinT patch_embed 第一层卷积改为多通道，复制预训练前 3 通道权重。"""
    backbone = model.backbone
    # Joiner 为 Sequential: [0] 是 SwinT，patch_embed 位于其内
    swin = backbone[0]
    old_conv = swin.patch_embed.proj
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        # 前 3 通道复用预训练 RGB 卷积权重
        new_conv.weight[:, :3, :, :] = old_conv.weight.data
        # 后 2 通道高斯初始化
        nn.init.normal_(new_conv.weight[:, 3:, :, :], mean=0.0, std=0.01)
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data
    swin.patch_embed.proj = new_conv
    swin.patch_embed.in_chans = in_channels
    return model


def load_model_3modal(feature_fusion: bool = False):
    """加载官方权重并改造为三模态输入。

    Args:
        feature_fusion: True 用特征层门控融合（9.2）；否则早期通道拼接（5 通道）。
    """
    args = SLConfig.fromfile(CFG_PATH)
    args.device = str(DEVICE)
    model = build_model(args)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    if feature_fusion:
        modify_model_feature_fusion(model)
    else:
        modify_model_first_conv(model, in_channels=5)
    model.eval().to(DEVICE)
    return model


# ----------------------------- 推理 -----------------------------
def predict_multimodal(model, image: torch.Tensor, caption: str):
    """三模态 5 通道输入 + 文本 -> (boxes_cxywh, logits, phrases)。

    降低收集阈值以保留多个候选框，供空间语义排序使用。
    """
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    t = image.unsqueeze(0).to(DEVICE)
    # 若模型被转为 FP16（--half），输入同步转换以避免 dtype 不匹配报错
    if next(model.parameters()).dtype != t.dtype:
        t = t.to(next(model.parameters()).dtype)
    with torch.no_grad():
        outputs = model(t, captions=[caption])
    logits = outputs["pred_logits"][0].sigmoid()
    boxes = outputs["pred_boxes"][0]
    mask = logits.max(dim=1)[0] > CANDIDATE_THRESHOLD
    boxes = boxes[mask]
    logits = logits[mask].max(dim=1)[0]
    return boxes, logits


def to_xyxy_norm(boxes_cxywh: torch.Tensor, h: int, w: int):
    """归一化 [cx,cy,w,h] -> [x1,y1,x2,y2] ∈ [0,1]（基于原图尺寸，考虑 resize 近似）。"""
    cx, cy, bw, bh = boxes_cxywh.unbind(dim=-1)
    x1 = (cx - bw / 2).clamp(0.0, 1.0)
    y1 = (cy - bh / 2).clamp(0.0, 1.0)
    x2 = (cx + bw / 2).clamp(0.0, 1.0)
    y2 = (cy + bh / 2).clamp(0.0, 1.0)
    box = torch.stack([x1, y1, x2, y2], dim=-1)
    return box.cpu().numpy()


def sanitize_box(b: np.ndarray) -> np.ndarray:
    """保证 x1<x2, y1<y2 且范围合法。"""
    b = b.astype(np.float64).copy()
    if b[0] >= b[2]:
        c = (b[0] + b[2]) / 2
        hw = 0.5
        b[0], b[2] = c - hw, c + hw
    if b[1] >= b[3]:
        c = (b[1] + b[3]) / 2
        hh = 0.5
        b[1], b[3] = c - hh, c + hh
    b[0:2] = np.maximum(b[0:2], 0.0)
    b[2:4] = np.minimum(b[2:4], 1.0)
    b[2] = max(b[2], b[0] + 1e-3)
    b[3] = max(b[3], b[1] + 1e-3)
    return b


# ----------------------------- 空间语义增强 -----------------------------
def parse_spatial_criterion(caption: str):
    """解析空间关键词，返回 (key, order, rank) 或 None。

    key: 'depth' | 'x' | 'y'；order: 'asc' | 'desc'（升/降序选优）；
    rank: 序数位次（1=最端/最近/最远，2=第二，-1=最后一个）。
    优先级：深度(远/近) > 水平(左/右) > 垂直(上/下)。
    """
    c = caption.lower().replace("-", " ").replace("_", " ")
    # 序数解析：first/second/third/.../last
    _ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
            "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
    rank = 1
    m = re.search(r"\b(first|second|third|fourth|fifth|sixth|seventh|"
                  r"eighth|ninth|tenth)\b", c)
    if m:
        rank = _ORD[m.group(1)]
    elif re.search(r"\blast\b", c):
        rank = -1
    # 深度：最远 / 最近
    if any(k in c for k in ["farthest", "furthest", "most distant",
                            "far from the camera", "farthest from",
                            "away from the camera", "the farthest"]):
        return "depth", "desc", rank
    if any(k in c for k in ["nearest", "closest", "closest to",
                            "nearest to", "most near", "close to the camera",
                            "near the camera", "the nearest"]):
        return "depth", "asc", rank
    # 水平：从右到左读序=右端；from left/right / far left/right
    if any(k in c for k in ["from right to left", "from the right",
                            "rightmost", "right side", "far right",
                            "to the right of", "on the right"]):
        return "x", "desc", rank
    if any(k in c for k in ["from left to right", "from the left",
                            "leftmost", "left side", "far left",
                            "to the left of", "on the left"]):
        return "x", "asc", rank
    # 垂直
    if any(k in c for k in ["topmost", "top of the", "above", "upper",
                            "highest", "on the top"]):
        return "y", "asc", rank
    if any(k in c for k in ["bottommost", "bottom of the", "below",
                            "lower", "lowest", "on the bottom"]):
        return "y", "desc", rank
    return None


def depth_at_box(depth_raw: np.ndarray, x1, y1, x2, y2, h, w, p=50):
    """取框内有效深度的 p 分位数；全部无效返回 None。"""
    x1i, x2i = int(x1 * w), max(int(x1 * w) + 1, int(x2 * w))
    y1i, y2i = int(y1 * h), max(int(y1 * h) + 1, int(y2 * h))
    x1i, x2i = max(0, x1i), min(w, x2i)
    y1i, y2i = max(0, y1i), min(h, y2i)
    roi = depth_raw[y1i:y2i, x1i:x2i].astype(np.float32)
    valid = (roi >= 300) & (roi <= 19999)
    if not valid.any():
        return None
    return float(np.percentile(roi[valid], p))


def strip_to_class(caption: str) -> str:
    """从空间 query 提炼类别名词，用于重新检测以扩充候选实例。

    仅对含空间词的 query 调用；无法可靠提炼时返回原句。
    """
    c = caption.lower().strip().rstrip(".")
    # 归一化连字符/下划线，保证 far-right -> far right 等被正确提炼
    c = c.replace("-", " ").replace("_", " ")
    for a in ("the ", "a ", "an "):
        if c.startswith(a):
            c = c[len(a):]
            break
    for w in ["first ", "second ", "third ", "fourth ", "fifth ", "last ",
              "next ", "middle ", "nearest ", "closest ", "farthest ",
              "furthest ", "leftmost ", "rightmost "]:
        c = c.replace(w, " ")
    for p in ["from the left", "from the right", "from left to right",
              "from right to left", "on the left", "on the right",
              "in the middle", "left side", "right side", "far left",
              "far right", "closest to the camera", "nearest to the camera",
              "farthest from the camera", "furthest from the camera",
              "most distant", "in the foreground", "in the background",
              "that is", "which is", "to the left of", "to the right of"]:
        c = c.replace(p, " ")
    c = " ".join(c.split())
    return c or caption


def extract_category(caption: str) -> str:
    """提炼用于空间重检测的短类别名词短语；无法提炼时退回 strip_to_class。

    原因：strip_to_class 对带丰富定语/从句的句子（如 'person in a pink
    shirt driving a red bumper car on the left background'）只能去掉少量
    固定介词短语，结果几乎仍是原句，重检测得不到更多同类实例。此处改为：
    - 句子含车辆名词 -> 直接返回车辆短语（如 'bumper car'），保证所有车入池；
    - 主语是人 -> 返回 'person'，扩充人的实例；
    - 否则回退 _supernoun_for / strip_to_class。
    """
    c = caption.lower().replace("-", " ").replace("_", " ")
    vpat = "|".join(re.escape(x) for x in sorted(VEHICLE_WORDS, key=len,
                                                 reverse=True))
    m = re.search(r"\b(" + vpat + r")(?:es|s)?\b", c)
    if m:
        return m.group(0)
    if re.search(r"\b(person|people|child|boy|girl|man|woman|men|women|lady|"
                 r"gentleman|driver)\b", c):
        return "person"
    supern = _supernoun_for(caption)
    if supern:
        return supern
    return strip_to_class(caption)


def spatial_select(boxes_xyxy: np.ndarray, logits: torch.Tensor,
                   depth_raw: np.ndarray, caption: str, h, w):
    """对候选框按空间语义排序选优，无空间词则取置信度最高框。

    boxes_xyxy: (K, 4) 归一化 [x1,y1,x2,y2] numpy。
    """
    if len(boxes_xyxy) == 0:
        return None
    bt = torch.as_tensor(boxes_xyxy, dtype=torch.float32, device=logits.device)
    keep = nms(bt, logits, 0.3)                     # 更严格去重，保留相邻独立实例
    bx = bt[keep].cpu().numpy()
    sc = logits[keep].cpu().detach().numpy()        # numpy 分数便于索引

    crit = parse_spatial_criterion(caption)
    if crit is None:
        return bx[int(np.argmax(sc))]

    key, order, rank = crit

    # 统计本 query 的候选框面积
    areas = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
    # 面积过滤：保留尺寸合理的候选（排除整图合并大框），若全为大框则全部保留
    small = np.where(areas <= MAX_BOX_AREA_FRAC)[0]
    pool = small if len(small) > 0 else np.arange(len(bx))
    # 仅当过滤后有足够候选可选时使用过滤池，否则用全部（防过度裁剪）
    pool = pool if len(pool) >= abs(rank) else np.arange(len(bx))

    def pick_by_key(vals: np.ndarray) -> np.ndarray:
        """在过滤后的候选里按空间值排序，取第 rank 个（rank=-1 取倒数第一）。"""
        vals = vals[pool]
        idx = np.argsort(vals)
        if order == "desc":
            idx = idx[::-1]
        pos = rank - 1 if rank > 0 else len(idx) + rank
        pos = int(np.clip(pos, 0, len(idx) - 1))
        return bx[pool[idx[pos]]]

    if key == "x":
        return pick_by_key((bx[:, 0] + bx[:, 2]) / 2)
    if key == "y":
        return pick_by_key((bx[:, 1] + bx[:, 3]) / 2)

    # 深度排序：已在 pool 基础上，按有效深度远近取第 rank 个
    pool_bx = bx[pool]
    stats = [(i, depth_at_box(depth_raw, *pool_bx[i].tolist(), h, w))
             for i in range(len(pool_bx))]
    present = [(i, d) for i, d in stats if d is not None]
    if not present:
        return bx[pool[int(np.argmax(sc[pool]))]]
    present.sort(key=lambda t: t[1], reverse=(order == "desc"))
    pos = rank - 1 if rank > 0 else len(present) + rank
    pos = int(np.clip(pos, 0, len(present) - 1))
    return pool_bx[present[pos][0]]


# --------------------- 属性语义增强（颜色视觉匹配） ---------------------
COLOR_WORDS = [
    "yellow", "orange", "purple", "violet", "brown", "green", "white",
    "black", "gray", "grey", "blue", "red", "pink", "cyan",
]


def parse_color_word(caption: str):
    """从 query 提取目标颜色属性词，返回规范颜色名；无则返回 None。

    简明用词边界匹配，连字符/下划线先归一为空格，保证 'yellow-tinted' 等可命中。
    取**最靠左**出现的颜色词：主目标名词短语（语法主语）通常位于句首，
    而定语从句/介词短语中的上下文颜色（如 'white deer behind a brown deer'
    的 brown、'blue car with a green shirt' 的 green）居后。按出现位置而非
    词表顺序取色，可避免多颜色词场景下张冠李戴。
    """
    c = caption.lower().replace("-", " ").replace("_", " ")
    best = None
    best_pos = None
    for color in COLOR_WORDS:
        for m in re.finditer(r"\b" + re.escape(color) + r"\b", c):
            if best_pos is None or m.start() < best_pos:
                best_pos = m.start()
                best = color
    return "gray" if best == "grey" else best


# --------------------- 车辆颜色解析（人/车混合 query 定位） ---------------------
VEHICLE_WORDS = [
    "bumper car", "car", "truck", "bus", "van", "suv", "sedan", "coupe",
    "jeep", "vehicle", "cart", "trolley", "carriage", "tractor", "lorry",
    "tram", "motorcycle", "motorbike", "scooter", "moped", "bike", "bicycle",
    "train",
]

# 人物处于车辆内的动词/介词（用于判断"人开车/坐车"类 query）
PERSON_IN_VEHICLE_WORDS = [
    "driving", "drives", "riding", "rides", "sitting in", "inside", "aboard",
]


def parse_vehicle_color(caption: str):
    """返回直接修饰车辆名词的颜色词（如 'red bumper car' -> 'red'）；无则 None。

    从车辆名词向前回溯一段"颜色短语"（颜色词 + and/or/冠词），取该短语
    **最靠左**的颜色作为主色。这样 'green and white car' 取 green（更特异），
    而 'person in a pink shirt driving a red car' 因 'pink' 被 'shirt driving'
    隔断、只回溯到 'red'，正确取到车辆颜色 red 而非衬衫颜色 pink。
    """
    c = caption.lower().replace("-", " ").replace("_", " ")
    vpat = "|".join(re.escape(x) for x in sorted(VEHICLE_WORDS, key=len,
                                                 reverse=True))
    for m in re.finditer(r"\b(" + vpat + r")(?:es|s)?\b", c):
        head = c[:m.start()]
        words = head.split()
        colors = []
        for w in reversed(words):
            w = w.strip(".,!?")
            if w in COLOR_WORDS:
                colors.append(w)
            elif w in ("and", "or", "&", "a", "an", "the", "of"):
                continue
            elif colors:
                # 已收集到颜色后遇到其它词，颜色短语到此结束
                break
            else:
                # 尚未遇到颜色，继续向前跨过 'driving' 等非颜色词
                continue
        if colors:
            col = colors[-1]                      # 颜色短语最靠左者
            return "gray" if col == "grey" else col
    return None


def parse_target_color(caption: str):
    """返回用于定位的目标颜色。

    对"主语是车辆"或"主语是人且处于车辆内"的 query，优先用车辆颜色
    （避免 'person in a pink shirt driving a red car' 误取衬衫粉 pink 而
    取车辆红 red）；否则退回 parse_color_word 的左起最左颜色。
    """
    c = caption.lower().strip().rstrip(".")
    c = re.sub(r"^(the|a|an)\s+", "", c)
    vpat = "|".join(re.escape(x) for x in sorted(VEHICLE_WORDS, key=len,
                                                 reverse=True))
    # 主语是车辆
    if re.match(r"^(?:" + vpat + r")(?:es|s)?\b", c):
        vc = parse_vehicle_color(c)
        if vc:
            return vc
    # 主语是人且处于车辆内
    if re.match(r"^(person|people|child|boy|girl|man|woman|men|women|lady|"
                r"gentleman|driver)\b", c):
        if any(k in c for k in PERSON_IN_VEHICLE_WORDS):
            vc = parse_vehicle_color(c)
            if vc:
                return vc
    return parse_color_word(caption)


def _color_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    """在 OpenCV HSV 空间返回目标颜色的像素掩码（bool）。

    H∈[0,179]、S∈[0,255]、V∈[0,255]。
    """
    h = hsv[..., 0].astype(np.int16)
    s = hsv[..., 1].astype(np.int16)
    v = hsv[..., 2].astype(np.int16)
    if color == "red":
        return (s >= 60) & (v >= 60) & ((h <= 12) | (h >= 168))
    if color == "orange":
        return (s >= 80) & (v >= 80) & (h >= 12) & (h < 25)
    if color == "yellow":
        return (s >= 80) & (v >= 100) & (h >= 20) & (h < 40)
    if color == "green":
        return (s >= 50) & (v >= 50) & (h >= 40) & (h < 85)
    if color == "cyan":
        return (s >= 50) & (v >= 50) & (h >= 85) & (h < 100)
    if color == "blue":
        return (s >= 50) & (v >= 50) & (h >= 100) & (h < 130)
    if color in ("purple", "violet"):
        return (s >= 50) & (v >= 50) & (h >= 130) & (h < 165)
    if color == "pink":
        return (v >= 130) & (s >= 30) & ((h <= 12) | (h >= 160))
    if color == "white":
        return (s <= 45) & (v >= 170)
    if color == "gray":
        return (s <= 45) & (v >= 60) & (v < 170)
    if color == "black":
        return (v <= 70)
    if color == "brown":
        return (s >= 50) & (v >= 40) & (v < 140) & (h >= 10) & (h < 30)
    return np.zeros(h.shape, dtype=bool)


def box_color_score(box, image_rgb: np.ndarray, color: str) -> float:
    """返回框内目标颜色像素占比 [0,1]。image_rgb 为原分辨率 RGB。"""
    x1, y1, x2, y2 = box
    h, w = image_rgb.shape[:2]
    x1i, x2i = int(x1 * w), max(int(x1 * w) + 1, int(x2 * w))
    y1i, y2i = int(y1 * h), max(int(y1 * h) + 1, int(y2 * h))
    x1i, x2i = max(0, x1i), min(w, x2i)
    y1i, y2i = max(0, y1i), min(h, y2i)
    roi = image_rgb[y1i:y2i, x1i:x2i]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    return float(_color_mask(hsv, color).mean())


def select_color_then_spatial(boxes_xyxy, logits, image_rgb, depth_raw,
                              caption, h, w, min_score=0.05, min_area=0.005):
    """综合颜色属性与空间语义选择最终框。

    有颜色词时先用颜色+面积**过滤**出候选子集，再在子集内做空间/置信度排序；
    颜色词全不匹配或面积不足时交回全量空间排序兜底。避免颜色与空间互斥导致
    误选（如"左侧绿车"按空间选到最左黄车、忽略了 green）。

    Args:
        boxes_xyxy: (K,4) 归一化候选框 [x1,y1,x2,y2]。
        logits: (K,) 各候选框检测置信度。
        image_rgb: 原分辨率 RGB numpy (h,w,3)。
        depth_raw: 原分辨率深度图，用于空间排序。
        caption: query 文本。
        h, w: 原图高宽。
        min_score: 目标颜色像素占比低于该阈值视为颜色不可见。
        min_area: 归一化面积低于该阈值的框视为过小干扰，排除。

    Returns:
        (4,) 归一化框，或 None。
    """
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    conf = torch.as_tensor(logits)
    if len(boxes) == 0:
        return None
    color = parse_target_color(caption)
    if color is not None:
        scores = np.array([box_color_score(b, image_rgb, color) for b in boxes])
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        # 颜色可见 + 面积充足才进入候选子集
        valid = (scores >= min_score) & (areas >= min_area)
        idx = np.where(valid)[0]
        if idx.size > 0:
            idx_t = torch.as_tensor(idx, dtype=torch.long, device=conf.device)
            return spatial_select(boxes[idx], conf[idx_t], depth_raw,
                                  caption, h, w)
        # 颜色全不匹配 -> 交回全量（漏斗兜底）
    return spatial_select(boxes, conf, depth_raw, caption, h, w)


# --------------------- 合并框扩充/切分（方案 1 & 2） ---------------------
def _box_pixel_aspect(box, h, w):
    """计算归一化框在像素空间的高宽比（>1 表示比宽更高）。"""
    x1, y1, x2, y2 = box
    pw = max((x2 - x1) * w, 1e-6)
    ph = max((y2 - y1) * h, 1e-6)
    return ph / pw


def _is_tall_entity_query(caption: str) -> bool:
    """判断 query 目标是否为本体高瘦的实体（人/树/柱/塔/门/墙/柜等）。

    这类目标的高瘦框即正确答案，不应触发候选扩充或高瘦框过滤。
    """
    c = caption.lower()
    # 允许单复数（trees/poles/posts 等）命中
    pat = r"\b(" + "|".join(TALL_ENTITY_WORDS) + r")(?:es|s)?\b"
    return re.search(pat, c) is not None


def _supernoun_for(caption: str):
    """返回 query 目标类别对应的上位词；未命中返回 None。

    用词边界正则匹配，避免子串误命中（如 "cat" 误中 "located"、'car' 误中 'carry'）。
    """
    c = caption.lower()
    for supern, keys in SUPERNOUN_MAP:
        # 允许单复数（motorcycles/scooters/rabbits 等）命中
        pat = r"\b(" + "|".join(re.escape(k) for k in keys) + r")(?:es|s)?\b"
        if re.search(pat, c):
            return supern
    return None


def expand_candidates(model, mm_t, boxes_norm, logits, query, h, w):
    """方案 1：检测到高瘦合并框时，用更细粒度通用名词重检测并并入子候选。

    高瘦框（人形长条）常把前景小物体与背后的真人合并；切换通用名词（如
    "toy"）可让 Grounding DINO 拆出独立实例。仅对非人形目标触发。
    """
    if len(boxes_norm) == 0:
        return boxes_norm, logits
    # 纯竖向实体（人/树等）跳过候选扩充；但"人+车"等混合 query 仍需拆出车辆
    # 候选（如 'person driving a red car' 需用 'vehicle' 重检测出左侧红车）。
    if _is_tall_entity_query(query) and _supernoun_for(query) != "vehicle":
        return boxes_norm, logits
    tall = any(_box_pixel_aspect(b, h, w) > TALL_BOX_ASPECT for b in boxes_norm)
    if not tall:
        return boxes_norm, logits

    # 变体词表：类名词 + 目标类别对应的上位词
    cls = strip_to_class(query)
    supern = _supernoun_for(query)
    variants = []
    for v in [cls, supern]:
        v = (v or "").strip().rstrip(".")
        if v and v not in variants:
            variants.append(v)
    for v in variants:
        if v.lower() == query.lower().strip().rstrip("."):
            continue  # 原 query 已检测过
        try:
            b2, l2 = predict_multimodal(model, mm_t, v)
        except Exception:  # noqa: BLE001
            continue
        if b2 is None or len(b2) == 0:
            continue
        n2 = to_xyxy_norm(b2, h, w)
        boxes_norm = np.concatenate([boxes_norm, n2], axis=0)
        logits = torch.cat([logits, l2])
    return boxes_norm, logits


def split_merged_by_depth(boxes_norm, logits, depth_full, h, w,
                          depth_gap=800, min_area_frac=0.01):
    """方案 2：对高瘦合并框按深度不连续切分，前景连通域作为独立子候选。

    前景物（更近）相对背景（更远）存在明显深度差时，按"比背景近 depth_gap"
    的像素连通域切出子框；只切分高瘦框，避免把正常物体拆碎。
    """
    if len(boxes_norm) == 0:
        return boxes_norm, logits
    new_boxes, new_scores = [], []
    for box in boxes_norm:
        if _box_pixel_aspect(box, h, w) <= TALL_BOX_ASPECT:
            continue
        subs = _split_box_by_depth(box, depth_full, h, w, depth_gap,
                                   min_area_frac)
        new_boxes.extend(s for s, _ in subs)
        new_scores.extend(s for _, s in subs)
    if not new_boxes:
        return boxes_norm, logits
    boxes_norm = np.concatenate([boxes_norm, np.array(new_boxes)], axis=0)
    extra = torch.tensor(new_scores, dtype=logits.dtype, device=logits.device)
    logits = torch.cat([logits, extra])
    return boxes_norm, logits


def _split_box_by_depth(box, depth_full, h, w, depth_gap, min_area_frac):
    """在单个高瘦框内按深度切分，返回 [(归一化子框, 置信度), ...]。"""
    x1, y1, x2, y2 = box
    x1i, x2i = int(x1 * w), max(int(x1 * w) + 1, int(x2 * w))
    y1i, y2i = int(y1 * h), max(int(y1 * h) + 1, int(y2 * h))
    x1i, x2i = max(0, x1i), min(w, x2i)
    y1i, y2i = max(0, y1i), min(h, y2i)
    if x2i <= x1i or y2i <= y1i:
        return []
    roi = depth_full[y1i:y2i, x1i:x2i].astype(np.float32)
    valid = (roi >= 300) & (roi <= 19999)
    if valid.sum() < 0.05 * roi.size:
        return []
    bg = float(np.percentile(roi[valid], 75))
    near = valid & (roi < (bg - depth_gap))
    if near.sum() < 0.02 * roi.size:
        return []
    near_u8 = near.astype(np.uint8)
    near_u8 = cv2.morphologyEx(near_u8, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    num, labels = cv2.connectedComponents(near_u8)
    rh, rw = roi.shape[:2]
    subs = []
    for cid in range(1, num):
        ys, xs = np.where(labels == cid)
        if len(ys) < 0.01 * roi.size:
            continue
        bx1 = xs.min() / rw * (x2 - x1) + x1
        bx2 = (xs.max() + 1) / rw * (x2 - x1) + x1
        by1 = ys.min() / rh * (y2 - y1) + y1
        by2 = (ys.max() + 1) / rh * (y2 - y1) + y1
        area = (bx2 - bx1) * (by2 - by1)
        if area < min_area_frac:
            continue
        subs.append(([bx1, by1, bx2, by2], 0.5))
    return subs


def filter_tall_boxes(boxes_norm, logits, query, h, w):
    """去掉高瘦合并框（仅当存在非高瘦候选且目标非竖向实体时）。

    否则无空间词的非竖向实体 query 会因 argmax 取到置信度更高的合并框而选错。
    """
    if len(boxes_norm) == 0 or _is_tall_entity_query(query):
        return boxes_norm, logits
    tall = np.array([_box_pixel_aspect(b, h, w) > TALL_BOX_ASPECT
                     for b in boxes_norm])
    non_tall = ~tall
    if non_tall.any():
        boxes_norm = boxes_norm[non_tall]
        mask = torch.as_tensor(non_tall, device=logits.device, dtype=torch.bool)
        logits = logits[mask]
    return boxes_norm, logits


def select_by_pose(pose_model, boxes_norm, logits, image_source, ir_path,
                   depth_full, target_pose, patch_size=224):
    """检测后姿态分类（方案 2）：对候选框分类朝向，选与 query 目标朝向最匹配的框。

    返回归一化 xyxy numpy (4,)；无法判断时返回 None，交由空间语义/置信度兜底。
    """
    if len(boxes_norm) == 0:
        return None
    bt = torch.as_tensor(boxes_norm, dtype=torch.float32, device=logits.device)
    keep = nms(bt, logits, 0.3)
    bx = bt[keep].cpu().numpy()
    if len(bx) == 0:
        return None
    ir_gray = load_ir_gray(ir_path)
    patches = []
    for box in bx:
        patches.append(prepare_patch(image_source, ir_gray, depth_full,
                                     box, size=patch_size))
    batch = torch.stack(patches, dim=0).to(DEVICE)
    with torch.no_grad():
        probs = pose_model(batch).softmax(dim=1)     # (K, 3)
    col = probs[:, target_pose]                       # 目标朝向概率
    best = int(col.argmax().item())
    return bx[best]


# ----------------------------- 主流程 -----------------------------
def main():
    global IMAGE_REF, DATA_ROOT  # noqa: PLW0603  （须在函数内首次引用前声明）
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=0, help="仅处理前 n 条测试")
    parser.add_argument("--test_json", type=str, default=str(DEFAULT_TEST_JSON),
                        help="问句清单（初赛 test.json 或 复赛 queries.json，结构同构）")
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT),
                        help="数据集根目录（图像路径的基准）")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--ref", type=int, default=None,
                        help="输入图像参考边（越小越快，默认 800）")
    parser.add_argument("--half", action="store_true",
                        help="GPU 上用 FP16 推理加速")
    parser.add_argument("--fusion", action="store_true",
                        help="用特征层门控融合（9.2）替代 5 通道早期拼接")
    parser.add_argument("--lora", type=str, default="",
                        help="LoRA 权重路径（9.1），加载后适配三模态/空间语义")
    parser.add_argument("--pose_ckpt", type=str, default="",
                        help="检测后姿态分类器权重路径（正/侧判别，方案 2）")
    args = parser.parse_args()

    # 固定随机种子，抑制跨进程/并发时的数值波动，保证边缘候选稳定入池
    torch.manual_seed(0)
    np.random.seed(0)

    if args.ref:
        IMAGE_REF = args.ref
    DATA_ROOT = Path(args.data_root)
    print(f"[INFO] 数据根目录 {DATA_ROOT}")
    print(f"[INFO] 问句清单   {args.test_json}")

    mode = "特征层门控融合" if args.fusion else "5 通道早期融合"
    print("=" * 60)
    print(f"加载 Grounding DINO + {mode} ...")
    print("=" * 60)
    model = load_model_3modal(feature_fusion=args.fusion)
    if args.lora:
        inject_lora(model, r=8, alpha=16.0)
        load_lora(model, args.lora)
        model.eval().to(DEVICE)
        print("[INFO] 已加载 LoRA 权重")
    pose_model = None
    pose_patch = 224
    if args.pose_ckpt:
        ck = torch.load(args.pose_ckpt, map_location="cpu", weights_only=True)
        pose_patch = int(ck.get("size", 224))
        base = int(ck.get("base", 32))
        pose_model = PoseClassifier(in_chans=5, num_classes=3, base=base).to(DEVICE)
        pose_model.load_state_dict(ck["state_dict"])
        pose_model.eval()
        print(f"[INFO] 已加载姿态分类器 {args.pose_ckpt} (patch={pose_patch})")
    if args.half and DEVICE.type == "cuda":
        model = model.half()                        # FP16 加速
        print("[INFO] 已开启 FP16 推理")
    print(f"[INFO] 模型就绪，设备 {DEVICE}")

    with open(args.test_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    ids = list(test_data.keys())
    if args.n > 0:
        ids = ids[:args.n]
    print(f"共 {len(ids)} 个待推理样本", flush=True)

    # 断点续跑：若输出文件已存在，跳过已完成的样本（支持中断后重启）
    results = {}
    if os.path.exists(args.out) and args.n == 0:
        try:
            with open(args.out, "r", encoding="utf-8") as f:
                existing = json.load(f)
            done = set(existing.keys()) & set(ids)
            if done:
                results = existing
                ids = [q for q in ids if q not in done]
                print(f"发现已有结果 {len(done)} 条，续跑剩余 {len(ids)} 条",
                      flush=True)
        except Exception:  # noqa: BLE001
            results = {}
    elif os.path.exists(args.out):
        # 测试模式也允许覆盖旧文件，但先清空避免与旧样本混淆
        results = {}

    t0 = time.time()
    last_save = t0
    for i, qid in enumerate(ids):
        rec = test_data[qid]
        try:
            vis_p = str(DATA_ROOT / rec["visible"])
            ir_p = str(DATA_ROOT / rec["infrared"])
            dep_p = str(DATA_ROOT / rec["depth"])
            image_source, mm_t, h, w, depth_full = load_multimodal_image(
                vis_p, ir_p, dep_p)
            boxes, logits = predict_multimodal(model, mm_t, rec["query"])
            boxes_norm = (to_xyxy_norm(boxes, h, w)
                          if len(boxes) > 0 else np.empty((0, 4)))

            # 空间 query：用类别名词重检测扩充候选实例池，便于排序区分
            if parse_spatial_criterion(rec["query"]) is not None:
                cls = extract_category(rec["query"])
                if cls.lower() != rec["query"].lower().strip().rstrip("."):
                    b2, l2 = predict_multimodal(model, mm_t, cls)
                    if len(b2) > 0:
                        n2 = to_xyxy_norm(b2, h, w)
                        if len(boxes_norm) > 0:
                            boxes_norm = np.concatenate([boxes_norm, n2], axis=0)
                            logits = torch.cat([logits, l2])
                        else:
                            boxes_norm, logits = n2, l2

            # 方案 1：高瘦合并框 -> 通用名词变体重检测扩充候选
            boxes_norm, logits = expand_candidates(
                model, mm_t, boxes_norm, logits, rec["query"], h, w)
            # 方案 2：高瘦合并框按深度不连续处切分
            boxes_norm, logits = split_merged_by_depth(
                boxes_norm, logits, depth_full, h, w)
            # 有更细小候选时，去掉高瘦合并框，避免 argmax 选错
            boxes_norm, logits = filter_tall_boxes(
                boxes_norm, logits, rec["query"], h, w)

            if len(boxes_norm) == 0:
                bbox = [0.25, 0.25, 0.75, 0.75]  # 兜底
            else:
                sel = None
                target_pose = parse_pose_word(rec["query"])
                if pose_model is not None and target_pose is not None:
                    sel = select_by_pose(pose_model, boxes_norm, logits,
                                         image_source, ir_p, depth_full,
                                         target_pose, patch_size=pose_patch)
                if sel is None:
                    # 颜色属性优先过滤，再在子集内做空间/置信度排序（颜色+空间可组合）
                    sel = select_color_then_spatial(
                        boxes_norm, logits, image_source, depth_full,
                        rec["query"], h, w)
                if sel is None:
                    bbox = [0.25, 0.25, 0.75, 0.75]
                else:
                    bbox = sanitize_box(sel).tolist()
            results[qid] = {
                "visible": rec["visible"],
                "infrared": rec["infrared"],
                "depth": rec["depth"],
                "query": rec["query"],
                "bbox": [round(float(v), 6) for v in bbox],
            }
        except Exception as e:  # noqa: BLE001
            results[qid] = {
                "visible": rec["visible"],
                "infrared": rec["infrared"],
                "depth": rec["depth"],
                "query": rec["query"],
                "bbox": [0.25, 0.25, 0.75, 0.75],
                "error": str(e),
            }
            print(f"[ERR] {qid}: {e}", flush=True)

        if (i + 1) % 20 == 0 or i + 1 == len(ids):
            el = time.time() - t0
            speed = (i + 1) / el
            eta = (len(ids) - i - 1) / speed if speed > 0 else 0
            print(f"  [{i+1}/{len(ids)}] speed={speed:.2f} img/s, "
                  f"elapsed={el:.0f}s, eta={eta/60:.0f}min", flush=True)

        # 每 100 条或每 10 分钟定期保存一次，防止长任务中断丢进度
        if (i + 1) % 100 == 0 or (time.time() - last_save) > 600:
            os.makedirs(Path(args.out).parent, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            last_save = time.time()
            print(f"  [checkpoint] 已保存 {len(results)} 条到 {args.out}",
                  flush=True)

    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {args.out}")
    print(f"总耗时: {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
"""通用 IO 工具。

注意：Windows 上 OpenCV 的 ``cv2.imread`` 无法处理含中文的路径，
因此统一使用 PIL / numpy.fromfile 读取图像，避免中文路径导致的读取失败。
"""
from __future__ import annotations

import json
import os
import zipfile
from typing import Any, Dict

import numpy as np
from PIL import Image


def read_image_rgb(path: str) -> np.ndarray:
    """读取图像为 RGB 的 uint8 numpy 数组 (H, W, 3)。

    使用 PIL，兼容中文路径。
    """
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def read_image_uint16(path: str) -> np.ndarray:
    """读取 16 位单通道图像（深度图）为 (H, W) uint16 数组。"""
    with Image.open(path) as img:
        arr = np.asarray(img)
    if arr.ndim == 3:  # 若被读成多通道，取第一通道
        arr = arr[..., 0]
    return arr.astype(np.uint16)


def load_json(path: str) -> Dict[str, Any]:
    """读取 UTF-8 编码的 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str) -> None:
    """写出 JSON（UTF-8，缩进 2，确保中文可读）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def zip_directory(src_dir: str, zip_path: str) -> str:
    """把目录内容打包为 zip（仅打包文件本身，不含顶层目录名）。

    注意：zip 输出路径不要位于 src_dir 内部，否则会把正在写入的 zip
    自身也打包进去，导致无限膨胀。
    """
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(src_dir):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, src_dir)
                if os.path.abspath(full) == os.path.abspath(zip_path):
                    continue  # 跳过 zip 自身
                zf.write(full, arcname=rel)
    return zip_path


def zip_file(src_path: str, zip_path: str, arcname: str | None = None) -> str:
    """把单个文件压缩为 zip。

    Args:
        src_path: 待压缩文件。
        zip_path: 输出 zip 路径。
        arcname: 包内文件名（默认使用源文件名）。
    """
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    arcname = arcname or os.path.basename(src_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(src_path, arcname=arcname)
    return zip_path

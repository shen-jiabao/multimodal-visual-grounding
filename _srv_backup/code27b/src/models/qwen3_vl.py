"""Qwen3-VL 零样本视觉定位封装（生成式 VLM）。

对应《模型适配方案-当前代码接入Qwen》（2026-09-23）：
    - 目标模型 Qwen3-VL-8B-Instruct（Apache-2.0；输出坐标体系 **[0,1000] 归一化**）；
    - 通过 chat template 输入（非短语拼接），关闭 thinking（``enable_thinking=False``）；
    - 输出解析 / 坐标换算在 :mod:`src.models.qwen3_vl_parse`（纯函数，可单测）；
    - 返回结构对齐 ``BaseGroundingModel``（``{scores, labels, boxes}``，boxes 为**像素坐标**），
      从而复用 matcher / 归一化 / 导出。

输出模式（config ``qwen.output_mode``）：
    - ``single_box``（默认，方案模式①）：逐查询一次生成，要求单个 bbox JSON；
    - ``multi_box``（方案模式③）：同图**一次**生成多个 ``{label, bbox_2d}``，保 matcher。
      **模式③使生成次数从"每查询一次"降到"每图一次"（本数据集 ~5690 → ~1295）。**

同图视觉特征缓存（config ``qwen.feature_cache``）：
    - Qwen3-VL 的 ``forward`` 仅在 ``pixel_values`` 非空时调用 ``get_image_features``（视觉塔），
      并对**只读**结果做 cat/masked_scatter。故缓存该输出、跳过视觉塔重算，其余（3D rope、
      scatter）仍走原生路径——**结果不变**。模式①下同图 N 条查询只需编码视觉塔一次。
    - 以图像内容哈希为键、有界 LRU；未找到 ``get_image_features`` 时自动关闭（安全降级）。

注意：本机未下载 Qwen 权重；端到端效果需在取得权重 + ≥12GB 显存 GPU 后，按方案阶段 III
冒烟验证（多分辨率坐标目检 + 解析成功率）。封装接口按 HF 官方用法 + 已装 5.16.1 源码编写。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.config import Config
from src.models.base import BaseGroundingModel
from src.models.qwen3_vl_parse import (
    build_multi_prompt,
    build_query_prompt,
    coords_1000_to_pixel,
    coords_1000_to_unit,
    infer_id_base,
    parse_multi_boxes_ex,
    parse_single_box,
    propagate_identical_queries,
    resolve_multi_target,
)
from src.postprocess.bbox import is_valid_bbox
from src.postprocess.search import aggregate_candidates
from src.utils.cache import LRUCache, content_hash

logger = logging.getLogger(__name__)


class Qwen3VLModel(BaseGroundingModel):
    """Qwen3-VL 生成式视觉定位封装（实现 :class:`BaseGroundingModel`）。"""

    def __init__(self, cfg: Config):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        super().__init__(cfg)
        logger.info("加载 Qwen3-VL：%s", cfg.model_path)
        self.processor = AutoProcessor.from_pretrained(cfg.model_path)

        # 半精度：Ada/Hopper 等优先 bf16（数值更稳），否则 fp16；非半精度为 fp32
        if cfg.model.use_half_precision:
            _bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            dtype = torch.bfloat16 if _bf16 else torch.float16
        else:
            dtype = torch.float32
        # 兼容 transformers 版本：4.x / 5.x 参数名为 torch_dtype / dtype
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                cfg.model_path, dtype=dtype
            )
        except TypeError:
            self.model = AutoModelForImageTextToText.from_pretrained(
                cfg.model_path, torch_dtype=dtype
            )
        self.model.to(self.device)
        self.model.eval()

        q = cfg.qwen
        self.output_mode = q.output_mode
        self.max_new_tokens = int(q.max_new_tokens)
        self.enable_thinking = bool(q.enable_thinking)
        # 模式③ ID 化输出（P0-2）：Targets 带序号，模型只回 id，归属无歧义
        # 缺省 True（ID 化）；用 getattr 读取，便于单测以 SimpleNamespace 桩件替换本类方法
        self.multi_box_id = bool(getattr(q, "multi_box_id", True))
        self.default_score = float(q.default_score)
        self.scale_x = float(q.coord_scale_x)
        self.scale_y = float(q.coord_scale_y)
        # 搜索式推理：候选采样 + 一致性投票（num_samples > 1 时生效）
        self.num_samples = max(1, int(getattr(q, "num_samples", 1)))
        self.sample_temperature = float(getattr(q, "sample_temperature", 0.7))
        self.sample_top_p = float(getattr(q, "sample_top_p", 0.9))
        self.vote_iou_threshold = float(getattr(q, "vote_iou_threshold", 0.5))
        self.vote_min_votes = max(1, int(getattr(q, "vote_min_votes", 1)))

        # 同图视觉特征缓存
        self.feature_cache = bool(getattr(q, "feature_cache", True))
        self._vision_cache: LRUCache = LRUCache(int(getattr(q, "cache_max_entries", 4)))
        self._active_image_key: Optional[str] = None
        self._orig_get_image_features = None
        if self.feature_cache:
            self._install_vision_cache()

        # 诊断采集（纯观测，不影响返回的预测）：每次 predict_boxes 调用重置
        # 用 getattr 读取，便于单测用 SimpleNamespace 桩件替换本类的方法
        self._collect_diag = bool(getattr(cfg.inference, "diag_dump", False))
        self._diag: Dict[str, Any] = {}
        self.reset_diag()

        logger.info(
            "Qwen3-VL 加载完成，设备=%s 精度=%s 模式=%s thinking=%s 视觉缓存=%s",
            self.device, dtype, self.output_mode, self.enable_thinking, self.feature_cache,
        )

    # ---------- 同图视觉特征缓存 ----------
    def _install_vision_cache(self) -> None:
        """把内层模型的 ``get_image_features`` 换为缓存包装（命中则跳过视觉塔）。"""
        inner = getattr(self.model, "model", None)
        if inner is None or not hasattr(inner, "get_image_features"):
            logger.warning("未找到 get_image_features，关闭视觉特征缓存")
            self.feature_cache = False
            return
        self._orig_get_image_features = inner.get_image_features
        cache = self._vision_cache

        def _cached(pixel_values, image_grid_thw=None, **kwargs):
            key = self._active_image_key
            if key is not None:
                entry = cache.get(key)
                if entry is not None and entry.get("output") is not None:
                    return entry["output"]
            out = self._orig_get_image_features(
                pixel_values, image_grid_thw=image_grid_thw, **kwargs
            )
            if key is not None:
                entry = cache.get(key)
                if entry is None:
                    entry = {}
                    cache.put(key, entry)
                entry["output"] = out
            return out

        inner.get_image_features = _cached
        logger.info("视觉特征缓存已启用（max_entries=%d）", cache.maxsize)

    # ---------- 诊断采集（纯观测；关闭时全部为空） ----------
    @property
    def _diag_on(self) -> bool:
        """是否采集诊断明细。

        用 getattr 兜底：单测用 ``SimpleNamespace`` 桩件直接绑定本类的方法
        （见 ``tests/test_qwen3_vl_search.py``），此时没有 ``_collect_diag``，
        取默认 False 即完全不采集，行为与改动前一致。
        """
        return bool(getattr(self, "_collect_diag", False))

    def reset_diag(self) -> None:
        """重置本次生成的诊断记录（每次 predict_boxes 调用前调用）。"""
        self._diag = {
            "prompt": None,           # 本次给模型的完整文本 prompt
            "images": 0,              # 喂入的图片张数（split 模式应为 3）
            "chunks": [],             # 每块的 query 列表（max_queries_per_prompt 分块）
            "replies": [],            # 模型原始回复（含被截断的）
            "pairs": [],              # 解析出的全部 (label, bbox_1000)——含匹配失败的
            "unmatched": [],          # match_label_to_query 返回 None 的 label
            "matched": 0,             # 匹配成功的 pair 数
            "dropped_invalid": 0,     # 坐标换算/合法性检查丢弃的 pair 数
            "candidates": {},         # num_samples>1 时：query -> 候选框(unit)列表
            "votes": {},              # num_samples>1 时：query -> 投票票数
            "id_base": 1,             # 模式③ ID 化：本条回复反推出的编号基准（0/1）
            "propagated": 0,          # 同文本 query 弱占优补齐的条数
        }

    # ---------- 生成 ----------
    def _generate(self, images: List[Image.Image], prompt: str,
                  num_return_sequences: int = 1) -> List[str]:
        """``images`` 为按序 PIL 图列表（图 1=可见光，其余为辅助模态图）。

        ``num_return_sequences`` > 1 时开启温度采样并返回多个候选回复
        （搜索式推理的候选生成）；默认 1 = 原来的贪心单次，行为不变。
        """
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        # 关闭 thinking（Instruct 版）；模型不支持该 kwarg 时忽略
        try:
            chat = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            chat = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = self.processor(text=[chat], images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        n = max(1, int(num_return_sequences))
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": n > 1,
        }
        if n > 1:
            gen_kwargs.update(
                num_return_sequences=n,
                temperature=self.sample_temperature,
                top_p=self.sample_top_p,
            )
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        gen = out[:, inputs["input_ids"].shape[1]:]
        replies = self.processor.batch_decode(gen, skip_special_tokens=True)
        return list(replies) if replies else [""]

    # ---------- 图像序列组装 ----------
    # 顺序固定：图 1 = 可见光（bbox 落在它上面），其后红外、深度。
    # 抽成独立方法是为了让「多变量同轮上云」时能逐变量离线重建：
    # 只要图像序列与图像键一致，同一份 diag 就能复算出各消融臂。
    _MODALITY_SPEC = (
        ("infrared", "infrared thermal image"),
        ("depth", "depth map (near=dark, far=bright; invalid regions=black)"),
    )

    def _assemble_images(self, image: np.ndarray, extra_images: Optional[Dict[str, np.ndarray]]
                         ) -> Tuple[List[Image.Image], List[str]]:
        """可见光在前，其后按固定顺序追加额外模态；返回 (PIL 图列表, 模态说明列表)。"""
        images: List[Image.Image] = [Image.fromarray(image)]
        modality_labels: List[str] = ["visible RGB image"]
        if extra_images:
            for name, label in self._MODALITY_SPEC:
                img = extra_images.get(name)
                if img is not None:
                    images.append(Image.fromarray(img))
                    modality_labels.append(label)
        return images, modality_labels

    def _image_key(self, image: np.ndarray,
                   extra_images: Optional[Dict[str, np.ndarray]]) -> Optional[str]:
        """视觉特征缓存键：含全部模态内容哈希，确保同图复用且不与他图串味。"""
        if not self.feature_cache:
            return None
        key_parts = [content_hash(image)]
        if extra_images:
            for name in sorted(extra_images):
                if extra_images[name] is not None:
                    key_parts.append(content_hash(extra_images[name]))
        return "|".join(key_parts)

    # ---------- 定位 ----------
    def predict_boxes(
        self,
        image: np.ndarray,
        texts: List[str],
        box_threshold: float = 0.20,
        text_threshold: float = 0.15,
        extra_images: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Dict[str, Any]]:
        """生成式定位。``box_threshold``/``text_threshold`` 对 Qwen 无对应语义（接口对齐）。

        ``extra_images``（``{"infrared": uint8(H,W,3), "depth": uint8(H,W,3)}``）非空时，
        以**多图对话**摄入三模态（图 1=可见光，其后红外/深度），替代单图融合。
        """
        if len(texts) == 0:
            return []
        # 组装图像序列：可见光在前，其后按固定顺序追加额外模态
        images, modality_labels = self._assemble_images(image, extra_images)
        # 图像键含全部模态，确保同图多查询复用视觉特征且不与他图串味（模式①）
        self._active_image_key = self._image_key(image, extra_images)

        h, w = image.shape[:2]
        # 诊断：一次 predict_boxes = 一张图（或一个分块）的一次调用，先重置
        if getattr(self, "_diag_on", False):
            self.reset_diag()
        if self.output_mode == "multi_box":
            labels, boxes, scores = self._predict_multi(images, modality_labels, texts, w, h)
        else:
            labels, boxes, scores = self._predict_single(images, modality_labels, texts, w, h)

        if not boxes:
            # 无可用框：诊断仍需保留（reply_len / unmatched 是判断"截断 vs 回显失败"的关键）
            return []
        return [{
            "scores": np.asarray(scores, dtype=np.float32),
            "labels": labels,
            "boxes": np.asarray(boxes, dtype=np.float32),
        }]

    def _accept(self, box1000, w: int, h: int) -> Optional[list]:
        """[0,1] 域合法性校验 → 转像素坐标；不合法返回 None。"""
        unit = coords_1000_to_unit(box1000, self.scale_x, self.scale_y)
        if not is_valid_bbox(unit):
            return None
        return coords_1000_to_pixel(box1000, w, h, self.scale_x, self.scale_y)

    # ---------- 候选投票（搜索式推理） ----------
    def _unit(self, box1000) -> Optional[list]:
        """[0,1000] 归一化坐标 -> [0,1] 单位框，不合法返回 None。"""
        unit = coords_1000_to_unit(box1000, self.scale_x, self.scale_y)
        if not is_valid_bbox(unit):
            return None
        return [float(v) for v in unit]

    def _vote(self, candidates: List[list]) -> Optional[Tuple[list, float]]:
        """多个候选框投票聚合为 (unit_box, 置信度)；置信度来自票数占比。"""
        if not candidates:
            return None
        if len(candidates) == 1:
            return ([float(v) for v in candidates[0]], self.default_score)
        res = aggregate_candidates(
            candidates,
            threshold=self.vote_iou_threshold,
            min_votes=min(self.vote_min_votes, len(candidates)),
        )
        if res is None:
            return None
        return (list(res["box"]), float(res["confidence"]))

    def _predict_single(self, images, modality_labels, texts, w, h
                        ) -> Tuple[List[str], List[list], List[float]]:
        """模式一：逐 query 生成；num_samples>1 时每 query 采样多候选并投票。"""
        if self.num_samples <= 1:
            labels: List[str] = []
            boxes: List[list] = []
            scores: List[float] = []
            for q in texts:
                prompt = build_query_prompt(q, modality_labels)
                reply = self._generate(images, prompt)[0]
                if getattr(self, "_diag_on", False):
                    self._diag["prompt"] = prompt
                    self._diag["images"] = len(images)
                    self._diag["chunks"].append([q])
                    self._diag["replies"].append(reply)
                box1000 = parse_single_box(reply)
                if getattr(self, "_diag_on", False):
                    self._diag["pairs"].append(
                        {"label": q, "box1000": None if box1000 is None else list(box1000),
                         "matched_query": q if box1000 is not None else None})
                if box1000 is None:
                    continue
                px = self._accept(box1000, w, h)
                if px is None:
                    if getattr(self, "_diag_on", False):
                        self._diag["dropped_invalid"] += 1
                    continue
                if getattr(self, "_diag_on", False):
                    self._diag["matched"] += 1
                labels.append(q)
                boxes.append(px)
                scores.append(self.default_score)
            return labels, boxes, scores

        labels = []
        boxes = []
        scores = []
        for q in texts:
            cands: List[list] = []
            prompt = build_query_prompt(q, modality_labels)
            for reply in self._generate(images, prompt, self.num_samples):
                if getattr(self, "_diag_on", False):
                    self._diag["replies"].append(reply)
                box1000 = parse_single_box(reply)
                if box1000 is None:
                    continue
                unit = self._unit(box1000)
                if unit is None:
                    continue
                cands.append(unit)
            if getattr(self, "_diag_on", False):
                self._diag["prompt"] = prompt
                self._diag["images"] = len(images)
                self._diag["chunks"].append([q])
                self._diag["candidates"][q] = [list(c) for c in cands]
            voted = self._vote(cands)
            if voted is None:
                continue
            unit, conf = voted
            if getattr(self, "_diag_on", False):
                self._diag["votes"][q] = {
                    "n_candidates": len(cands),
                    "voted": list(unit),
                    "confidence": conf,
                }
            labels.append(q)
            boxes.append([unit[0] * w, unit[1] * h, unit[2] * w, unit[3] * h])
            scores.append(conf)
        return labels, boxes, scores

    def _predict_multi(self, images, modality_labels, texts, w, h
                       ) -> Tuple[List[str], List[list], List[float]]:
        """模式二：同图一次生成多个框；num_samples>1 时多次采样后按 query 投票。"""
        if self.num_samples <= 1:
            labels: List[str] = []
            boxes: List[list] = []
            scores: List[float] = []
            prompt = build_multi_prompt(
                texts, modality_labels, use_id=getattr(self, "multi_box_id", False))
            reply = self._generate(images, prompt)[0]
            if getattr(self, "_diag_on", False):
                self._diag["prompt"] = prompt
                self._diag["images"] = len(images)
                self._diag["chunks"].append(list(texts))
                self._diag["replies"] = [reply]
            pairs = parse_multi_boxes_ex(reply)
            # 编号基准（1-based / 0-based）只能从**整条回复的 ID 全集**反推：
            # 逐条兼容会把 id=0 与 id=1 都映到 queries[0]（碰撞），比严格 1-based
            # 更糟。判 0-based 的三条保守规则见 infer_id_base。
            base = infer_id_base([t for t, _, _ in pairs], len(texts))
            for tid, label, box1000 in pairs:
                q = resolve_multi_target(tid, label, texts, id_base=base)
                if getattr(self, "_diag_on", False):
                    self._diag["pairs"].append(
                        {"target_id": tid, "label": label,
                         "box1000": list(box1000), "matched_query": q})
                if q is None:
                    if getattr(self, "_diag_on", False):
                        self._diag["unmatched"].append(label)
                    continue
                px = self._accept(box1000, w, h)
                if px is None:
                    if getattr(self, "_diag_on", False):
                        self._diag["dropped_invalid"] += 1
                    continue
                labels.append(q)
                boxes.append(px)
                scores.append(self.default_score)
            # 同图**逐字相同**的 query：ID 化提示词下模型可能只回一个 id，
            # 另一条落空 → 占位框。把已有框复制给同文本兄弟是弱占优（见
            # propagate_identical_queries）。不动任何已存在的框。
            got = list(zip(labels, boxes))
            for q, px in propagate_identical_queries(texts, got):
                if q not in labels:
                    labels.append(q)
                    boxes.append(px)
                    scores.append(self.default_score)
            if getattr(self, "_diag_on", False):
                self._diag["id_base"] = base
                self._diag["propagated"] = max(0, len(labels) - len(got))
                self._diag["matched"] = len(labels)
            return labels, boxes, scores

        # 多候选：按 query 收集所有样本给出的框，再投票（保持首现顺序）
        pool: Dict[str, List[list]] = {}
        order: List[str] = []
        prompt = build_multi_prompt(
            texts, modality_labels, use_id=getattr(self, "multi_box_id", False))
        for reply in self._generate(images, prompt, self.num_samples):
            if getattr(self, "_diag_on", False):
                self._diag["replies"].append(reply)
            # 同 num_samples<=1：编号基准由整条回复的 ID 全集反推
            pairs = parse_multi_boxes_ex(reply)
            base = infer_id_base([t for t, _, _ in pairs], len(texts))
            for tid, label, box1000 in pairs:
                q = resolve_multi_target(tid, label, texts, id_base=base)
                if getattr(self, "_diag_on", False):
                    self._diag["pairs"].append(
                        {"target_id": tid, "label": label,
                         "box1000": list(box1000), "matched_query": q})
                if q is None:
                    if getattr(self, "_diag_on", False):
                        self._diag["unmatched"].append(label)
                    continue
                unit = self._unit(box1000)
                if unit is None:
                    if getattr(self, "_diag_on", False):
                        self._diag["dropped_invalid"] += 1
                    continue
                if q not in pool:
                    pool[q] = []
                    order.append(q)
                pool[q].append(unit)

        if getattr(self, "_diag_on", False):
            self._diag["prompt"] = prompt
            self._diag["images"] = len(images)
            self._diag["chunks"].append(list(texts))
            self._diag["candidates"] = {k: [list(v) for v in vs]
                                        for k, vs in pool.items()}

        labels = []
        boxes = []
        scores = []
        for q in order:
            voted = self._vote(pool[q])
            if getattr(self, "_diag_on", False):
                self._diag["votes"][q] = {
                    "n_candidates": len(pool[q]),
                    "voted": None if voted is None else list(voted[0]),
                    "confidence": None if voted is None else voted[1],
                }
            if voted is None:
                continue
            unit, conf = voted
            labels.append(q)
            boxes.append([unit[0] * w, unit[1] * h, unit[2] * w, unit[3] * h])
            scores.append(conf)
        return labels, boxes, scores

"""主模型：多模态视觉定位（Multimodal Visual Grounding）。

输入：可见光 + 红外 + 深度图像 + 英文查询
输出：归一化边界框 [x1, y1, x2, y2]
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from config import MODEL_CFG
from models.encoders import MultiModalImageEncoder, TextEncoder
from models.fusion import CrossModalFusion


class BBoxHead(nn.Module):
    """MLP 边界框预测头。"""

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 4),
        )
        # 偏置初始化：让初始预测偏向图像中心
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用 sigmoid 让输出在 [0, 1]，但内部表示 [x1,y1,x2,y2]
        raw = self.mlp(x)
        cxcywh = raw.chunk(2, dim=-1)[0]  # 占位
        # 实际：直接输出 xyxy 偏移，最后用 sigmoid 归一化
        # 把输出拆成 cxcy 和 wh，再过 sigmoid
        cx = torch.sigmoid(raw[..., 0:1])
        cy = torch.sigmoid(raw[..., 1:2])
        w = torch.sigmoid(raw[..., 2:3]) * 0.5 + 0.05  # 限制最大占比 0.5
        h = torch.sigmoid(raw[..., 3:4]) * 0.5 + 0.05
        x1 = (cx - 0.5 * w).clamp(0, 1)
        y1 = (cy - 0.5 * h).clamp(0, 1)
        x2 = (cx + 0.5 * w).clamp(0, 1)
        y2 = (cy + 0.5 * h).clamp(0, 1)
        return torch.cat([x1, y1, x2, y2], dim=-1)


class MultimodalGroundingModel(nn.Module):
    """多模态视觉定位主模型。"""

    def __init__(self, cfg: dict | None = None) -> None:
        super().__init__()
        cfg = cfg or MODEL_CFG
        self.cfg = cfg

        # 文本编码器
        self.text_encoder = TextEncoder(
            model_name=cfg["text_encoder"],
            pretrained=cfg["text_pretrained"],
            max_length=cfg["text_max_length"],
        )
        text_dim = self.text_encoder.hidden_size

        # 视觉多模态编码器
        self.image_encoder = MultiModalImageEncoder(
            backbone=cfg["image_backbone"],
            pretrained=cfg["image_pretrained"],
            out_dim=cfg["image_feature_dim"],
        )
        vis_dim = cfg["image_feature_dim"]

        # 统一到融合维度
        self.text_align = nn.Linear(text_dim, cfg["fusion_dim"])
        self.vis_align = nn.Linear(vis_dim, cfg["fusion_dim"])

        # 跨模态融合
        self.fusion = CrossModalFusion(
            dim=cfg["fusion_dim"],
            num_layers=cfg["fusion_layers"],
            num_heads=cfg["fusion_heads"],
            dropout=cfg["fusion_dropout"],
        )

        # BBox 头
        self.bbox_head = BBoxHead(cfg["fusion_dim"])

    def forward(
        self,
        rgb: torch.Tensor,
        ir: torch.Tensor,
        depth: torch.Tensor,
        texts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            rgb: (B, 3, H, W)
            ir: (B, 3, H, W)
            depth: (B, 3, H, W)
            texts: List[str] 长度 B
        Returns:
            dict: 'bbox' (B, 4) 归一化 xyxy
        """
        # 文本
        t_out = self.text_encoder(texts)
        text_seq = t_out["last_hidden_state"]  # (B, T, Dt)
        # 对齐
        text_seq = self.text_align(text_seq)

        # 视觉
        v_out = self.image_encoder(rgb, ir, depth)
        vis_spatial = v_out["spatial"]  # (B, N, Dv)
        vis_global = v_out["global"]   # (B, Dv)
        vis_spatial = self.vis_align(vis_spatial)
        vis_global = self.vis_align(vis_global)

        # 计算文本 mask（非 padding 位置）
        if self.text_encoder.pretrained:
            enc = self.text_encoder.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.text_encoder.max_length,
                return_tensors="pt",
            )
            text_mask = enc["attention_mask"].to(text_seq.device)
        else:
            # 离线模式：非 0 即有效
            text_mask = (text_seq.abs().sum(dim=-1) > 0).long()

        # 融合
        fused = self.fusion(
            text_seq=text_seq,
            text_mask=text_mask,
            visual_spatial=vis_spatial,
            visual_global=vis_global,
        )
        bbox = self.bbox_head(fused)
        return {"bbox": bbox, "fused": fused}


def build_model(cfg: dict | None = None) -> MultimodalGroundingModel:
    """工厂函数。"""
    return MultimodalGroundingModel(cfg or MODEL_CFG)

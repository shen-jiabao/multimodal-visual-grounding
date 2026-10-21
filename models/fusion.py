"""跨模态融合：文本与多模态视觉特征交互。

设计：使用多层 Transformer Decoder 块（Pre-LN），文本作为 query，视觉作为 key/value。
最终将 [CLS] 文本 token + 全局视觉 + 空间池化特征拼接过 MLP 解码到 bbox。
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class CrossModalFusion(nn.Module):
    """跨模态融合模块。

    输入：
        text_seq: (B, T, D)
        visual_spatial: (B, N, D)
        visual_global: (B, D)
    输出：
        fused: (B, D) 融合后特征。
    """

    def __init__(
        self,
        dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.dim = dim
        # 文本维度到融合维度的投影（若需要）
        self.text_proj = nn.Linear(dim, dim) if dim != dim else nn.Identity()
        self.visual_proj = nn.Linear(dim, dim) if dim != dim else nn.Identity()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 融合后的 FFN
        self.fuse = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        text_seq: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        visual_spatial: torch.Tensor | None = None,
        visual_global: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            text_seq: (B, T, D)
            text_mask: (B, T) bool/int，True 表示 padding。Transformer 默认 key_padding_mask=True 即 padding。
            visual_spatial: (B, N, D)
            visual_global: (B, D)
        """
        text_q = self.text_proj(text_seq)
        if visual_spatial is not None:
            visual_kv = self.visual_proj(visual_spatial)
        else:
            visual_kv = visual_global.unsqueeze(1)  # (B, 1, D)

        # key_padding_mask：True 表示忽略该位置
        kpm = None
        if text_mask is not None:
            kpm = ~text_mask.bool()  # 1 表示 padding

        fused_text = self.decoder(
            tgt=text_q, memory=visual_kv, memory_key_padding_mask=None, tgt_key_padding_mask=kpm
        )
        # 池化：用 attention mask 做加权平均
        if text_mask is not None:
            m = text_mask.unsqueeze(-1).float()
            text_pool = (fused_text * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            text_pool = fused_text.mean(dim=1)
        # 拼接视觉全局
        if visual_global is None:
            visual_global = visual_kv.mean(dim=1)
        fused = self.fuse(torch.cat([text_pool, visual_global], dim=-1))
        return fused

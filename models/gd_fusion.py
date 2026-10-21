# -*- coding: utf-8 -*-
"""三模态特征层融合（替换输入通道拼接）。

思想：不再把 RGB/红外/深度简单按通道 cat 成 5 通道再过一个卷积（早期融合），
而是让三种模态各自经过独立的 patch_embed，得到各自的 patch token 序列，
再通过可学习门控加权融合，最后送入 SwinT 共享的 stage 层。

收益：深度/红外以可学习方式作用于定位，尤其改善“远近”判断。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from groundingdino.models.GroundingDINO.backbone.swin_transformer import (
    PatchEmbed,
)


class GatedMultimodalPatchEmbed(nn.Module):
    """三模态独立 patch 嵌入 + 门控融合。

    输入 `x` 为 5 通道张量 (B, 5, H, W)：前 3 通道 RGB、第 4 通道红外、
    第 5 通道深度。输出形状与 SwinT 原始 patch_embed 完全一致 (B, C, Wh, Ww)，
    因此可直接替换 SwinTransformer.patch_embed，后续 pos_drop/layers 不变。
    """

    def __init__(
        self,
        patch_size: int = 4,
        embed_dim: int = 96,
        norm_layer=None,
    ) -> None:
        super().__init__()
        # 三个独立 patch 嵌入：RGB 复用预训练权重，红外/深度单通道
        self.rgb_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )
        self.ir_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=1,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )
        self.depth_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=1,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )
        # 模态门控权重：初始偏向 RGB，红外/深度由微调逐步激活
        self.gate = nn.Parameter(torch.tensor([3.0, 0.0, 0.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 5, H, W) -> (B, C, Wh, Ww)。"""
        rgb = x[:, :3]
        ir = x[:, 3:4]
        depth = x[:, 4:5]
        f_rgb = self.rgb_embed(rgb)
        f_ir = self.ir_embed(ir)
        f_depth = self.depth_embed(depth)
        weight = torch.softmax(self.gate, dim=0)  # (3,)
        fused = weight[0] * f_rgb + weight[1] * f_ir + weight[2] * f_depth
        return fused


def modify_model_feature_fusion(model: nn.Module) -> nn.Module:
    """将 SwinT 单一 5 通道 patch_embed 替换为三模态门控融合模块。

    前提：模型已 `load_state_dict`（此时 patch_embed 仍为 3 通道预训练权重）。
    RGB 分支复用原始 3 通道卷积，红外/深度单通道卷积用 RGB 通道均值初始化。
    """
    swin = model.backbone[0]
    old_embed = swin.patch_embed

    embed_dim = old_embed.embed_dim
    patch_size = old_embed.patch_size
    norm_layer = type(old_embed.norm) if old_embed.norm is not None else None

    fusion = GatedMultimodalPatchEmbed(
        patch_size=patch_size,
        embed_dim=embed_dim,
        norm_layer=norm_layer,
    )

    with torch.no_grad():
        # RGB 分支复用原始预训练 patch_embed（含 proj 与 norm）
        fusion.rgb_embed = old_embed
        # 红外/深度单通道卷积用 RGB 3 通道权重的中值初始化，保留统计先验
        w = old_embed.proj.weight.data  # (out, 3, kh, kw)
        mean_w = w.mean(dim=1, keepdim=True)  # (out, 1, kh, kw)
        fusion.ir_embed.proj.weight.data.copy_(mean_w)
        fusion.depth_embed.proj.weight.data.copy_(mean_w)
        if old_embed.proj.bias is not None:
            fusion.ir_embed.proj.bias.data.copy_(old_embed.proj.bias.data)
            fusion.depth_embed.proj.bias.data.copy_(old_embed.proj.bias.data)

    swin.patch_embed = fusion
    swin.patch_embed.in_chans = 5
    return model
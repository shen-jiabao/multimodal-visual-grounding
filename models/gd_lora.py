# -*- coding: utf-8 -*-
"""Grounding DINO 的 LoRA 低秩适配。

对 SwinT 骨干与 BERT 文本编码器的 `nn.Linear` 层注入低秩适配器，
冻结基座、只训练 LoRA 参数（<1% 参数量），用于适配三模态输入与空间语义。

公式：`h = W0 x + (B A) x * (alpha / r)`，A∈R^{r×in}、B∈R^{out×r}。
"""
from __future__ import annotations

import re
from typing import Dict, List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """包裹原始 nn.Linear 的低秩适配层。

    原权重冻结，forward 输出 = 原线性输出 + 低秩增量。
    """

    def __init__(
        self,
        linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        in_features = linear.in_features
        out_features = linear.out_features
        # A 用小高斯、B 用零初始化，初始增量为 0，不破坏预训练表达
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 冻结原层
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        delta = self.dropout(x @ self.lora_A.T @ self.lora_B.T) * self.scale
        return base + delta


# 目标层名模式：仅注入 SwinT 注意力/MLP/patch-merge 与 BERT 的注意力/前馈
_LORA_TARGETS = (
    r"\.attn\.qkv$",
    r"\.attn\.proj$",
    r"\.mlp\.fc[12]$",
    r"\.reduction$",
    r"\.query$",
    r"\.key$",
    r"\.value$",
    r"\.intermediate\.dense$",
    r"\.output\.dense$",
)


def _is_target(name: str) -> bool:
    """判断该 Linear 层是否为 LoRA 注入目标。"""
    return any(re.search(p, name) is not None for p in _LORA_TARGETS)


def inject_lora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> nn.Module:
    """递归将目标 nn.Linear 替换为 LoRALinear，冻结基座。"""
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _is_target(name):
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, LoRALinear(module, r, alpha, dropout))
            replaced += 1
    print(f"[LoRA] 注入 {replaced} 个线性层 (r={r}, alpha={alpha})")
    return model


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """仅提取 LoRA 参数与门控参数（用于保存）。"""
    out: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name or "gate" in name:
            out[name] = p.detach().cpu()
    return out


def load_lora(model: nn.Module, path: str) -> nn.Module:
    """加载 LoRA 权重（严格按名匹配，忽略缺失）。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd = ckpt.get("lora", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(sd, strict=False)
    print(f"[LoRA] 已加载权重: {path}")
    return model


def freeze_except_lora(model: nn.Module) -> List[nn.Parameter]:
    """冻结所有参数，仅保留 LoRA/门控参数可训练，返回可训练参数列表。"""
    trainable: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name or "gate" in name:
            p.requires_grad_(True)
            trainable.append(p)
        else:
            p.requires_grad_(False)
    return trainable
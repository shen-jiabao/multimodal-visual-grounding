"""多模态编码器：文本（DistilBERT/CLIP 文本）、视觉（timm ResNet/ViT）。"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def _try_import_timm():
    try:
        import timm
        return timm
    except ImportError as e:  # pragma: no cover
        raise ImportError("缺少 timm 库，请先 `pip install timm`") from e


def _try_import_transformers():
    try:
        from transformers import AutoModel, AutoTokenizer
        return AutoModel, AutoTokenizer
    except ImportError as e:  # pragma: no cover
        raise ImportError("缺少 transformers 库，请先 `pip install transformers`") from e


# ---------------- 文本编码器 ----------------
class TextEncoder(nn.Module):
    """基于 HuggingFace 的文本编码器（默认 distilbert-base-uncased）。

    离线/无网络时，可设置 pretrained=False 自动用一个小型 LSTM 兜底。

    输出：每个 token 的 (B, T, D) 序列特征及 pooled (B, D) 向量。
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        pretrained: bool = True,
        max_length: int = 32,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.pretrained = pretrained
        self.model_name = model_name
        if pretrained:
            AutoModel, AutoTokenizer = _try_import_transformers()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=cache_dir
            )
            self.model = AutoModel.from_pretrained(
                model_name, cache_dir=cache_dir
            )
            self.hidden_size = self.model.config.hidden_size
        else:
            # 离线兜底：一个轻量 word embedding + BiLSTM
            self.tokenizer = None
            self.model = None
            vocab_size = 30522  # 与 bert-base 默认 vocab 一致
            embed_dim = 128
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                embed_dim, embed_dim // 2, num_layers=1,
                batch_first=True, bidirectional=True,
            )
            self.proj = nn.Linear(embed_dim, embed_dim)
            self.hidden_size = embed_dim

    def forward(self, texts: list[str]) -> Dict[str, torch.Tensor]:
        """编码文本。

        Args:
            texts: 长度 B 的字符串列表。
        Returns:
            dict: 'last_hidden_state' (B, T, D), 'pooled' (B, D)。
        """
        if not self.pretrained:
            # 离线模式：使用哈希 token
            max_len = self.max_length
            device = self.embed.weight.device
            ids_list = [_simple_tokenize(t, max_len) for t in texts]
            ids = torch.tensor(ids_list, dtype=torch.long, device=device)
            mask = (ids != 0).long()
            emb = self.embed(ids)
            seq, _ = self.lstm(emb)
            seq = self.proj(seq)
            pooled = (seq * mask.unsqueeze(-1).float()).sum(1) / mask.sum(1, keepdim=True).clamp(min=1).float()
            return {"last_hidden_state": seq, "pooled": pooled}

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(next(self.model.parameters()).device) for k, v in enc.items()}
        out = self.model(**enc)
        # 序列特征
        seq = out.last_hidden_state  # (B, T, D)
        # 池化：attention mask 加权平均
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (seq * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return {"last_hidden_state": seq, "pooled": pooled}


def _simple_tokenize(text: str, max_len: int) -> list[int]:
    """离线兜底 tokenizer：基于字符哈希，简单分词。"""
    # 拆成小写单词
    import re
    tokens = re.findall(r"[a-z0-9']+", text.lower())[:max_len]
    out = [1]  # CLS
    for tok in tokens:
        # 用稳定哈希映射到 vocab
        h = (hash(tok) & 0x7FFFFFFF) % (30522 - 2) + 2
        out.append(h)
    # padding
    while len(out) < max_len:
        out.append(0)
    return out[:max_len]


# ---------------- 视觉编码器 ----------------
class ImageEncoder(nn.Module):
    """基于 timm 的视觉编码器（默认 resnet18）。

    输出：空间特征 (B, D, h, w) 与全局特征 (B, D)。
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        in_chans: int = 3,
        out_dim: int = 512,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        timm = _try_import_timm()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,           # 去掉分类头
            global_pool="avg",
            features_only=False,
        )
        # 获取原始特征维度
        feat_dim = self.backbone.num_features
        # 提取中间层用于空间特征（以 resnet18 为例：layer4 输出 512 通道）
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        # 投影到统一维度
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )
        # 空间特征提取钩子：layer4
        self._feat_maps: Dict[str, torch.Tensor] = {}

        def hook(_m, _i, output):
            self._feat_maps["feat"] = output

        # 注册 hook
        if hasattr(self.backbone, "layer4"):
            self.backbone.layer4.register_forward_hook(hook)
        elif hasattr(self.backbone, "stages"):  # timm 新结构（如 convnext）
            self.backbone.stages[-1].register_forward_hook(hook)
        elif hasattr(self.backbone, "stages_4"):
            self.backbone.stages_4.register_forward_hook(hook)
        else:
            # 退而求其次：仅使用全局特征
            self._feat_maps = {}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """输入 (B, 3, H, W)，输出空间特征与全局特征。"""
        # 触发 hook 提取中间层
        global_feat = self.backbone(x)  # (B, feat_dim)
        if "feat" in self._feat_maps:
            fmap = self._feat_maps.pop("feat")  # (B, C, h, w)
        else:
            # 没有空间特征可用时，构造 1x1 空间特征
            fmap = global_feat.unsqueeze(-1).unsqueeze(-1)
        # 空间特征降采样到固定 7x7 池化，便于融合
        b, c, h, w = fmap.shape
        # 全局平均池化 + 投影
        global_proj = self.proj(global_feat)  # (B, out_dim)
        # 空间特征投影
        fmap_flat = fmap.flatten(2).transpose(1, 2)  # (B, hw, C)
        fmap_proj = self.proj(fmap_flat)            # (B, hw, out_dim)
        return {
            "global": global_proj,    # (B, out_dim)
            "spatial": fmap_proj,     # (B, hw, out_dim)
        }


# ---------------- 视觉多模态聚合 ----------------
class MultiModalImageEncoder(nn.Module):
    """对 RGB / IR / Depth 三种模态各自使用一个 ImageEncoder，再聚合。

    - 简单加权融合：global 用 concat+线性；spatial 用相加（带权重）。
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        out_dim: int = 512,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.rgb_encoder = ImageEncoder(
            backbone=backbone,
            pretrained=pretrained,
            in_chans=3,
            out_dim=out_dim,
            cache_dir=cache_dir,
        )
        self.ir_encoder = ImageEncoder(
            backbone=backbone,
            pretrained=pretrained,
            in_chans=3,
            out_dim=out_dim,
            cache_dir=cache_dir,
        )
        self.depth_encoder = ImageEncoder(
            backbone=backbone,
            pretrained=pretrained,
            in_chans=3,        # 深度已复制到 3 通道
            out_dim=out_dim,
            cache_dir=cache_dir,
        )
        # 模态权重（可学习）
        self.gate = nn.Sequential(
            nn.Linear(out_dim * 3, 3),
            nn.Softmax(dim=-1),
        )
        self.out_dim = out_dim

    def forward(
        self,
        rgb: torch.Tensor,
        ir: torch.Tensor,
        depth: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """输入三模态图像张量，输出 global 特征与 spatial 特征。"""
        rgb_out = self.rgb_encoder(rgb)
        ir_out = self.ir_encoder(ir)
        depth_out = self.depth_encoder(depth)
        # 1) global：拼接后门控
        global_cat = torch.cat(
            [rgb_out["global"], ir_out["global"], depth_out["global"]], dim=-1
        )  # (B, 3*out_dim)
        gate = self.gate(global_cat)  # (B, 3)
        global_feat = (
            gate[:, 0:1] * rgb_out["global"]
            + gate[:, 1:2] * ir_out["global"]
            + gate[:, 2:3] * depth_out["global"]
        )
        # 2) spatial：加权和
        s = (
            rgb_out["spatial"]
            + ir_out["spatial"]
            + depth_out["spatial"]
        ) / 3.0
        return {"global": global_feat, "spatial": s}

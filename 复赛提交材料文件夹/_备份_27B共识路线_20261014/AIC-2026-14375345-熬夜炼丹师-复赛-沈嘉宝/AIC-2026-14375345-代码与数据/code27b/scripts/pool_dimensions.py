#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""投票池的配置规则维度说明。

投票池 `consensus_pool.json.gz`（32 套）由本工程在**不同配置组合**下的多次
全量运行输出汇总而成。本文件枚举当前代码中实际存在的可配置维度及其取值域，
作为池构成的规则说明与复现参照。

维度定义与 `src/config.py` 一一对应；用 `python scripts/pool_dimensions.py`
可打印本表。
"""
from __future__ import annotations


# 维度表：名称 / 取值域 / 本工程提交时的取值 / 说明
DIMENSIONS = [
    {
        "name": "model.name_or_path",
        "domain": ["8B", "27B", "32B", "72B", "其他规模"],
        "used": "27B（主件与参照件）",
        "note": "基座模型规模。池内成员覆盖多档规模，主件与参照件均为 27B",
    },
    {
        "name": "model.use_half_precision",
        "domain": ["True (bfloat16)", "False (float32)"],
        "used": "True",
        "note": "推理精度；bf16 带来浮点级解码波动",
    },
    {
        "name": "inference.fusion_mode",
        "domain": ["none", "low_light", "gated"],
        "used": "low_light",
        "note": "多模态融合模式：不融合 / 低光红外增强 / 逐像素门控三模态融合",
    },
    {
        "name": "inference.multimodal_input_mode",
        "domain": ["split", "fused"],
        "used": "split",
        "note": "输入组织：三模态分图送入 / 融合为伪 RGB 单图",
    },
    {
        "name": "inference.enable_multimodal_fusion",
        "domain": ["True", "False"],
        "used": "True",
        "note": "是否启用多模态融合链路",
    },
    {
        "name": "inference.low_light_threshold",
        "domain": ["数值（默认 60.0）"],
        "used": "60.0",
        "note": "低光场景判定阈值（可见光平均亮度）",
    },
    {
        "name": "inference.enable_depth_prior",
        "domain": ["True", "False"],
        "used": "False",
        "note": "深度先验重排序（仅 closest/farthest 且深度置信 high/medium 时生效）",
    },
    {
        "name": "inference.chain_enforce",
        "domain": ["off", "sort", "project"],
        "used": "off",
        "note": "序数链一致性修正：关闭 / 排序重指派 / 投影修正",
    },
    {
        "name": "qwen.output_mode",
        "domain": ["single_box", "multi_box"],
        "used": "multi_box",
        "note": "输出模式：逐查询单框 / 同图多框",
    },
    {
        "name": "qwen.multi_box_id",
        "domain": ["True", "False"],
        "used": "True",
        "note": "ID 化输出：模型只回序号（True）还是回显整句标签（False）",
    },
    {
        "name": "qwen.enable_thinking",
        "domain": ["True", "False"],
        "used": "False",
        "note": "是否输出思维链（关闭以控制生成长度）",
    },
    {
        "name": "qwen.max_new_tokens",
        "domain": ["512", "1024", "其他"],
        "used": "1024",
        "note": "单次生成上限；过低会导致回复截断丢框",
    },
    {
        "name": "qwen.num_samples",
        "domain": ["1", ">1"],
        "used": "1",
        "note": "候选采样数；>1 时开启温度采样并按 IoU 投票聚合",
    },
    {
        "name": "qwen.feature_cache",
        "domain": ["True", "False"],
        "used": "True",
        "note": "同图视觉特征缓存（同图多查询只编码一次）",
    },
    {
        "name": "inference.max_queries_per_prompt",
        "domain": ["1 ~ 20"],
        "used": "20",
        "note": "单次提示词容纳的查询数，影响同图批处理粒度",
    },
]


def main() -> int:
    print("投票池的配置规则维度（与 src/config.py 对应）\n")
    for i, d in enumerate(DIMENSIONS, 1):
        dom = " | ".join(str(x) for x in d["domain"])
        print(f"{i:>2}. {d['name']}")
        print(f"    取值域  ：{dom}")
        print(f"    本工程用：{d['used']}")
        print(f"    说明    ：{d['note']}\n")
    print(
        "投票池由上述维度的不同取值组合、各执行一次全量推理后汇总而成；\n"
        "池内保存的是各次运行的 bbox 坐标，不含任何人工修改。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""配置加载模块。

负责读取 `configs/config.yaml`，解析为带默认值的配置对象，并支持
命令行参数的浅层覆盖（仅覆盖顶层已存在字段）。

用法：
    from src.config import load_config
    cfg = load_config("configs/config.yaml", overrides={"model.device": "cpu"})
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional

import yaml

# 项目根目录（src/config.py 的上一级再上一级）
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)


@dataclass
class DataConfig:
    root: str = "dataset"
    queries_file: str = "queries/queries.json"
    visible_dir: str = "Images/visible"
    infrared_dir: str = "Images/infrared"
    depth_dir: str = "Images/depth"


@dataclass
class ModelConfig:
    type: str = "qwen3_vl"
    name_or_path: str = "checkpoints/Qwen3-VL-8B-Instruct"
    device: str = "auto"
    use_half_precision: bool = False
    box_threshold: float = 0.20
    text_threshold: float = 0.15


@dataclass
class InferenceConfig:
    max_queries_per_prompt: int = 20
    enable_multimodal_fusion: bool = True
    low_light_threshold: float = 60.0
    # 多模态融合模式（选型建议 §二 技术路线）：none | low_light | gated
    #   low_light = 现有基线（低光红外增强，行为不变）
    #   gated     = 逐像素门控三模态融合（阶段二 模态适配 + 阶段三 GatedFusion）
    fusion_mode: str = "low_light"
    # gated 模式下的门控参数（训练无关启发式，非按标注调参）
    fusion_ir_weight: float = 1.0
    fusion_depth_weight: float = 1.0
    fusion_gate_scale: float = 30.0
    fallback: str = "none"
    # 深度先验重排序（仅 closest/farthest 且深度 high/medium 时启用）
    enable_depth_prior: bool = False
    # 多模态输入模式：split（可见光+红外+深度分模态多图，供 VLM）| fused（伪 RGB 融合单图）
    multimodal_input_mode: str = "split"
    # 序数链修正（跨查询一致性约束，零样本）：off | sort | project
    chain_enforce: str = "off"
    chain_gap_ratio: float = 0.0
    # 单框最大平移量（归一化坐标，0 = 不限）
    chain_max_shift: float = 0.0
    # 诊断落盘（纯观测，不影响预测结果）：把模型原始回复 / 解析出的
    # (label, bbox) / 匹配结果 / 候选投票明细写成 JSONL，供离线归因。
    # 背景：复赛归档只留了每 query 的最终框，"512 token 截断"与
    # "label 回显失败导致 match_label_to_query 返回 None"在归档上不可区分。
    diag_dump: bool = False
    # 诊断输出目录（相对项目根）；文件名为 diag_raw_<split或run>.jsonl
    diag_dir: str = "outputs/diag"


@dataclass
class QwenConfig:
    """Qwen3-VL 封装配置（模型适配方案-当前代码接入Qwen §2.5）。"""
    # 输出模式：single_box（逐查询单框）| multi_box（同图多框）
    output_mode: str = "multi_box"
    # 单次生成上限。512 → 1024：diag 全量归因证实 92 条丢框全部是回复截断
    # （truncated_json 81 + truncated_budget 8 + 其他 3），echo_fail = 0；
    # 经验式 need ≈ 512×n/k 最大 805，1024 全覆盖；未截断的块按 EOS 提前停，零额外前向。
    max_new_tokens: int = 1024
    # 关闭 thinking（Instruct 版；Thinking 版需设 False 或接受更长生成）
    enable_thinking: bool = False
    # 模式③ ID 化输出（P0-2）：Targets 带序号，要求模型只回 {"id": n, "bbox_2d": [...]}，
    # 不再回显整句 label。输出更短（缓解 multi_box 截断）、归属无歧义（不再依赖子串/词重叠匹配）。
    # False = 旧行为（回显整句 label），可用于复现 09-24 基线做 A/B。
    multi_box_id: bool = True
    # Qwen 无原生检测置信度：统一赋值（供 matcher 回退使用）
    default_score: float = 1.0
    # [0,1000] 坐标非均匀缩放修正系数（默认 1.0；阶段 III 目检后按需微调并写单测）
    coord_scale_x: float = 1.0
    coord_scale_y: float = 1.0
    # 同图视觉特征缓存（跳过视觉塔重算；模式①下同图 N 查询只编码一次）
    feature_cache: bool = True
    cache_max_entries: int = 4
    # ---- 搜索式推理：候选采样 + 一致性投票 ----
    # >1 时开启温度采样，每 query 生成 num_samples 个候选并投票聚合
    num_samples: int = 1
    sample_temperature: float = 0.7
    sample_top_p: float = 0.9
    vote_iou_threshold: float = 0.5
    vote_min_votes: int = 1


@dataclass
class OutputConfig:
    dir: str = "outputs"
    predictions_json: str = "predictions.json"
    submission_zip: str = "submission.zip"


@dataclass
class PostprocessConfig:
    """后处理配置：无效/缺失预测的中性占位框等。"""
    fallback_bbox: list[float] = field(
        default_factory=lambda: [0.49, 0.49, 0.51, 0.51]
    )


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    qwen: QwenConfig = field(default_factory=QwenConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)

    # ---------- 便捷路径拼接 ----------
    @property
    def data_root(self) -> str:
        return os.path.join(PROJECT_ROOT, self.data.root)

    @property
    def queries_path(self) -> str:
        return os.path.join(self.data_root, self.data.queries_file)

    @property
    def model_path(self) -> str:
        """模型路径解析。

        支持四种写法：
            - 绝对路径；
            - ``checkpoints/`` 下的本地子目录（相对项目根，如 ``checkpoints/Qwen3-VL-8B-Instruct``）；
            - 项目根目录下的本地目录名；
            - HuggingFace 在线 ID（含 ``/``，如 ``Qwen/Qwen3-VL-8B-Instruct``）。

        本地目录统一返回基于 PROJECT_ROOT 的绝对路径，避免依赖当前工作目录。
        """
        p = self.model.name_or_path
        if os.path.isabs(p):
            return p
        # 含路径分隔符：先按 PROJECT_ROOT 下本地子目录解析，不存在则视为 HF 在线 ID
        if "/" in p or "\\" in p:
            local = os.path.join(PROJECT_ROOT, p.replace("/", os.sep))
            return local if os.path.isdir(local) else p
        # 简单目录名：按 PROJECT_ROOT 下解析
        return os.path.join(PROJECT_ROOT, p)

    @property
    def output_dir(self) -> str:
        return os.path.join(PROJECT_ROOT, self.output.dir)


def _set_nested(cfg: Config, dotted_key: str, value: Any) -> None:
    """按 ``section.field`` 形式覆盖配置项。"""
    parts = dotted_key.split(".")
    obj: Any = cfg
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def load_config(path: Optional[str] = None,
                overrides: Optional[Dict[str, Any]] = None) -> Config:
    """加载 YAML 配置并应用覆盖项。

    Args:
        path: YAML 文件路径；为空时使用默认的 ``configs/config.yaml``。
        overrides: ``{"section.field": value}`` 形式的覆盖字典。

    Returns:
        配置对象。
    """
    if path is None:
        path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")

    cfg = Config()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        _merge(cfg, raw)

    if overrides:
        for k, v in overrides.items():
            _set_nested(cfg, k, v)

    return cfg


def _merge(cfg: Config, raw: Dict[str, Any]) -> None:
    """将 YAML 顶层 section 合并到 dataclass 中（仅更新已存在字段）。"""
    for section_name, values in raw.items():
        if not hasattr(cfg, section_name) or not isinstance(values, dict):
            continue
        section = getattr(cfg, section_name)
        valid_fields = {f.name for f in fields(section)}
        for k, v in values.items():
            if k in valid_fields:
                setattr(section, k, v)

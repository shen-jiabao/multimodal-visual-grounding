"""
全局配置：路径、超参、模型选项
Global configuration: paths, hyperparameters, model options
"""
from pathlib import Path
import torch

# ----------------------------- 路径配置 -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "初赛数据集-基于大模型的多模态视觉理解与推理"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
WORK_DIR = PROJECT_ROOT / "work_dirs"
for _d in (CHECKPOINT_DIR, WORK_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------- 数据集配置 -----------------------------
# 标注 JSON 文件名（初赛/复赛可能改名）
DEFAULT_TRAIN_JSON = "train.json"
DEFAULT_VAL_JSON = "val.json"
DEFAULT_TEST_JSON = "test.json"

# 三种视觉模态在 JSON 中的字段名
MODALITY_KEYS = {
    "visible": "visible",    # 可见光 RGB
    "infrared": "infrared",  # 红外（3 通道但本质单通道灰度）
    "depth": "depth",        # 深度 16-bit 单通道
}
TEXT_KEY = "query"
BBOX_KEY = "bbox"

# ----------------------------- 训练超参 -----------------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 训练
TRAIN_CFG = dict(
    epochs=80,
    batch_size=8,
    num_workers=4,
    lr=3e-4,
    weight_decay=1e-4,
    warmup_epochs=1,
    grad_clip=1.0,
    log_interval=20,
    save_top_k=3,
    # 损失权重
    lambda_l1=1.0,
    lambda_iou=3.0,
    lambda_giou=2.0,
)

# ----------------------------- 模型配置 -----------------------------
MODEL_CFG = dict(
    # 文本编码器
    text_encoder="bert-base-uncased",         # HuggingFace 模型名
    text_max_length=40,
    text_pretrained=True,                     # 离线时改为 False，会用简单 LSTM 兜底
    text_dim=768,
    # 视觉编码器
    image_backbone="resnet50",                # timm 中的 backbone 名称
    image_pretrained=True,                    # 离线时改为 False
    image_feature_dim=1024,                   # 三种模态特征统一维度
    # 融合
    fusion_layers=4,
    fusion_heads=8,
    fusion_dim=1024,
    fusion_dropout=0.15,
    # 适配
    use_pretrained_clip=False,                # True 时使用 CLIP（需联网）
)

# ----------------------------- 预处理配置 -----------------------------
# 输入图像尺寸（H, W）—— 竞赛标准 448x448
IMAGE_SIZE = (448, 448)

# RGB 归一化（ImageNet）
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)

# 红外：先转灰度再 3 通道复制
# 深度：16-bit，clip 到 [DEPTH_MIN, DEPTH_MAX]，再除以 19999 归一化
DEPTH_MIN_MM = 300
DEPTH_MAX_MM = 19999


# ----------------------------- 评估配置 -----------------------------
EVAL_CFG = dict(
    iou_threshold=0.5,   # ACC@0.5
)


def get_device() -> str:
    """获取当前可用设备。"""
    return DEVICE

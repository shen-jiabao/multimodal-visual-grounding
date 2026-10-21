# -*- coding: utf-8 -*-
"""构建复赛提交材料包（一次性、可重跑、可审计）。

为什么用脚本而不是手工复制：20 个代码文件都硬编码了本机绝对路径
（`d:\\Desktop\\2026AiC`），手工改必漏；脚本可**逐一报告改动**，且可重跑比对。

打包只做一件事的改动：把本机绝对路径换成跨机可解析的 `ROOT`
（`AIC_ROOT` 环境变量优先，缺省取 `code/` 的上一级）。**算法逻辑一行未改**。

用法
    python -B _build_submission_pkg.py
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
ARCH = R / "semifinal_remote_archive" / "root"
PKG = R / "复赛提交材料"
CODE = PKG / "code"

# 进包的脚本：(目标名, 源文件)
FILES = [
    # 共享引擎与路径库
    ("_fuse_qwen.py", R / "_fuse_qwen.py"),
    ("_fuse_semi.py", R / "_fuse_semi.py"),
    ("_sub_path.py", R / "_sub_path.py"),
    # 数据预处理
    ("_analyze_ordinal3.py", R / "_analyze_ordinal3.py"),
    ("_survey_modality.py", R / "_survey_modality.py"),
    # 序数层确定性规则
    ("_variant_det_semi.py", R / "_variant_det_semi.py"),
    ("_variant_depth_semi.py", R / "_variant_depth_semi.py"),
    # 枚举重跑与合并
    ("_enum_merge.py", R / "_enum_merge.py"),
    ("_enum_retry.py", ARCH / "_enum_retry.py"),
    # GroundingDINO 三模态备选分支（实测不佳，仅存档）
    ("_gd_enum.py", R / "_gd_enum.py"),
    ("_gd_merge.py", R / "_gd_merge.py"),
    ("predict_gd_3modal.py", R / "predict_gd_3modal.py"),
    # 主模型推理（GPU）
    ("_qwen_full_pipeline.py", ARCH / "_qwen_full_pipeline.py"),
    ("_probe_models.py", ARCH / "_probe_models.py"),
    ("_smoke_qwen32b.py", ARCH / "_smoke_qwen32b.py"),
    # 装配、读数、校验
    ("_assembly_32b.py", R / "_assembly_32b.py"),
    ("_assembly_ord_prime.py", R / "_assembly_ord_prime.py"),
    ("_assembly_ord_som.py", R / "_assembly_ord_som.py"),
    ("_ord_layer_levers.py", R / "_ord_layer_levers.py"),
    ("_measure_zero.py", R / "_measure_zero.py"),
    ("_d_measure.py", R / "_d_measure.py"),
    ("_survey_modality_semi.py", R / "_survey_modality_semi.py"),
    ("_validate_semi.py", R / "_validate_semi.py"),
    ("_test_assembly.py", R / "_test_assembly.py"),
    # ★跨代换模型路线（预登记 I → 线上 0.7963，即当前交付件）
    ("_q38_gate.py", ARCH / "_q38_gate.py"),          # G2/G3 冒烟（含目检图生成）
    ("_measure_zero_sup646_q38.py", R / "_measure_zero_sup646_q38.py"),
    ("_q38_next_step.py", R / "_q38_next_step.py"),
    # ★异质模型路线（预登记 L / N）——当前交付件出自 N
    ("_k_scope429.py", R / "_k_scope429.py"),      # 作用域切分（复现装配器分桶）
    ("_k_scale_diag.py", R / "_k_scale_diag.py"),  # 坐标空间判据（H1/H2/H3）
    ("_viz_q38.py", R / "_viz_q38.py"),            # 改动面可视化（框 + 查询原文）
    # 远端通道（工程与运维）
    ("_rc.py", R / "_rc.py"),
    ("_up_semi.py", R / "_up_semi.py"),
]

# ★远端 shell 脚本：**原样复制，不做 Python 路径注入**
#   理由：`inject()` 会插入 `import os/ROOT` 头，插进 bash 脚本即语法错误。
#   这些脚本内的路径本来就是远端绝对路径（/root/rivermind-data），跨机可解析性
#   由 `Q38_*` 环境变量提供，不需要改写。放在 code/remote_shell/ 下以示区分。
SHELL = [
    ("_get_serve_q38.sh", ARCH / "_get_serve_q38.sh"),   # 起服（含 --disable-custom-all-reduce）
    ("_launch_q38.sh", ARCH / "_launch_q38.sh"),         # 全量启动器（五道闸门）
    ("_q38_gate.sh", ARCH / "_q38_gate.sh"),             # 闸门冒烟包装
    ("_q38_poll.sh", ARCH / "_q38_poll.sh"),             # 就绪轮询
    ("_q38_dl.sh", ARCH / "_q38_dl.sh"),                 # 权重下载（断点续传 + 逐片 sha256）
    # ★异质模型路线（K 件）——当前交付件出自该路线
    ("_k_serve.sh", ARCH / "_k_serve.sh"),                 # K 起服启动器（含目录名断言）
    ("_k_poll.sh", ARCH / "_k_poll.sh"),                   # K 就绪轮询（以 HTTP 为准）
    ("_k_gate.sh", ARCH / "_k_gate.sh"),                   # 闸门冒烟（含坐标越界判据）
    ("_k_fill_poll.sh", ARCH / "_k_fill_poll.sh"),         # 批量进度查询（参数化，可复用）
    ("_k_fill_launch.sh", ARCH / "_k_fill_launch.sh"),     # L 件（429 位）启动
    ("_k_n_launch.sh", ARCH / "_k_n_launch.sh"),           # N 件（519 位）启动
]

# 中间产物：算法自产（非赛方数据），放在包根以便脚本里的相对路径直接命中。
# 覆盖"序数核心词表 → 8B 两阶段输出 → 离线规则层 → 枚举合并"全链所需的输入，
# 使离线规则层与校验**无需 GPU 即可复跑**。总量约 3 MB。
ARTIFACTS = [
    "_ordinal_cores_r2.json",
    "_enum_raw.json",
    "_enum_targets.json",
    "_semi_qwen3vl_all.json",       # 8B 直接定位（序数层推算的候选来源）
    "_semi_qwen3vl_som_all.json",   # 8B 序数 SoM 选中结果
    "_semi_retry_ir.json",          # 补框链第 2 级（换模态）
    "_semi_retry_thermal.json",
    "_semi_retry19.json",
    "_semi_swapA33.json",           # 红外对照实验存档
    "_semi_swap58.json",
    "_semi_nobox_qids.json",
    "_semi_thermal_qids.json",
    "_semi_rest19_qids.json",
    "_gd_enum_raw.json",            # GroundingDINO 备选分支的原始候选
    # 三模态路线（预登记 D）的作用范围与小样清单：使 `_d_measure.py` 的判据
    # **不传参数即可复算**（它的 --sample 默认指向 _d_sample300.json）。
    "_d_scope.json",                # 作用范围 1652 条（主类型 ∈ {距离/远近, 热/光照} 的非序数）
    "_d_sample300.json",            # 分层随机小样 300 条（测量前冻结）
    # ★cross-generation 路线（预登记 I）复算所需
    "_vlx_sup_split.json",          # 646「空间最高级」层清单（分层读数与测量件构造用）
    "_qwen32b_direct_ord.json",     # 32B AWQ 直接定位（全 5690）⇒ 序数层作用域判据的对照源
    # ★异质模型路线（预登记 L / N）复算所需 —— 使 L / N 两件**无需 GPU 即可离线复算**
    "_qwen38_direct.json",          # q38 直出（全 5690）⇒ 分桶与作用域切分的输入
    "_k72b_direct_519.json",        # N 件产物（519 位）⇒ 装配出交付件
    "_k72b_direct_429.json",        # L 件产物（429 位，负面结果，存档）
    "_k_scope429.json",             # L 的作用域清单
    "_k_scope519.json",             # N 的作用域清单
    "_k_gate20.json",               # 20 条冒烟的原始输出 ⇒ 坐标空间判据（H1/H2/H3）的证据
]

# 规则层复跑所需的基线件（保留工作区的"按得分归目录"结构，_sub_path 可直接找到）
BASELINES = [
    ("0.7459/result_final_semi.json", "0.7459/result_final_semi.json"),
    ("0.7578/result_semi_ord_det_ss_y_qfb.json",
     "0.7578/result_semi_ord_det_ss_y_qfb.json"),
    # 规则层脚本以 `result_semi_enum_trunc` 为基线名解析（`_measure_zero` /
    # `_assembly_32b` / `_test_assembly`），故这一版必须在场。
    ("0.7598/result_semi_enum_trunc.json",
     "0.7598/result_semi_enum_trunc.json"),
    # 当前最高件也要在此按名可解析（`_assembly_ord_prime` 以 0.7742 件的名字
    # `result_semi_n32_ordbase` 为基线，故必须同时在场）。
    ("0.7742/result_semi_n32_ordbase.json",
     "0.7742/result_semi_n32_ordbase.json"),
    ("0.7766/result_semi_n32_orddirect.json",
     "0.7766/result_semi_n32_orddirect.json"),
    # ★0.7786 件是 0.7963 件的**直接基线**：`_assembly_32b.py --base
    #   result_semi_n32bf16_nonord` 就是用它做替换底座的，必须在场才能复算交付件。
    ("0.7786/result_semi_n32bf16_nonord.json",
     "0.7786/result_semi_n32bf16_nonord.json"),
    # ★交付件本体也必须按名可解析（`_measure_zero_sup646_q38.py` /
    #   `_q38_next_step.py` 都以 `result_semi_q38_nonord` 为基线名解析）。
    ("0.7963/result_semi_q38_nonord.json",
     "0.7963/result_semi_q38_nonord.json"),
    # ★当前交付件（N 件）所在归档位。得分**尚未产生**，故目录名为「待提交」；
    #   拿到线上得分后应改名为该分数（与既有惯例一致），并同步本表与 RESULT_*。
    ("待提交/result_semi_k72b_n519.json",
     "待提交/result_semi_k72b_n519.json"),
]

RESULT_DIR = R / "submission_zips" / "semifinal"
# ★结果文件 = **排行榜上对应的结果文件** = 当前交付件。
#   ★★ 当前为 `result_semi_k72b_n519`（预登记 N：K 件只动 q38 改动过的 519 位），
#      **线上得分尚未产生**（未提交）⇒ RESULT_SCORE 记「待定」。
#      已实测最高分是 **0.7963**（`result_semi_q38_nonord`），仍作为基线件保留在
#      `0.7963/`；若 N 的实测分 ≤ 0.7963，应把本组常量改回
#      `0.7963` / `result_semi_q38_nonord` / `"0.7963"` 并重跑本脚本。
RESULT_BEST = RESULT_DIR / "待提交"
RESULT_STEM = "result_semi_k72b_n519"
RESULT_SCORE = "待定（未提交）"

# 训练与评估脚手架（初赛期开发、**未用于本提交**）。放进包是为了让"数据预处理 → 模型训练
# → 预测推理 → 评估"四个环节在代码上都有对应实现，同时由 README 明确说明本方案最终
# 未采用任何训练权重（收益全部来自提示工程与确定性规则）。
TRAINING = [
    "prepare_data.py",            # 数据准备与校验
    "train.py",                   # 训练入口
    "train_competition.py",       # 无 bbox 标注场景的训练
    "train_lora_gd.py",           # GroundingDINO LoRA 低秩微调
    "train_pose_classifier.py",   # 检测后姿态分类器
    "predict.py",                 # 推理入口（输出官方 JSON）
    "config.py",                  # 全局配置
    "evaluate.py",                # ACC@0.5 / mIoU 评估
]

HEADER = '''# ===== 提交包路径适配（打包时自动注入；算法逻辑未改动）=====
import os as _os
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
ROOT = _Path(_os.environ.get("AIC_ROOT") or _HERE.parent)
ROOT_S = str(ROOT)
# ===== 注入结束 =====
'''

FUT = "from __future__ import annotations"


def inject(text: str) -> tuple[str, str]:
    """把路径适配头注入到合法位置，返回 (新文本, 所用规则说明)。"""
    lines = text.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        if ln.strip() == FUT:
            return "".join(lines[:i + 1]) + HEADER + "".join(lines[i + 1:]), \
                   f"接在 `{FUT}` 之后"
    # 兜底：接在开头首个 import/from 块的最后一行之后（对单行 docstring 也稳）
    last = None
    for i, ln in enumerate(lines[:80]):
        if re.match(r"^(import |from )", ln):
            last = i
    if last is not None:
        return "".join(lines[:last + 1]) + HEADER + "".join(lines[last + 1:]), \
               "接在首个 import 块之后"
    raise RuntimeError("找不到安全的注入位置")


SUBS = [
    (re.compile(r'r"d:\\Desktop\\2026AiC\\'), 'str(ROOT) + r"\\'),
    (re.compile(r'r"d:\\Desktop\\2026AiC"'), 'ROOT_S'),
    (re.compile(r'Path\(r"d:\\Desktop\\2026AiC"\)'), 'ROOT'),
    (re.compile(r"r'd:\\Desktop\\2026AiC'"), 'ROOT_S'),
]


def patch(name: str, text: str) -> tuple[str, list[str]]:
    """替换本机绝对路径；返回 (新文本, 改动说明)。"""
    log = []
    # ★同目录脚本必须用 _HERE 定位，不能用 ROOT：包内 ROOT 是包根、脚本在 code/。
    #   这一条是真跑 `_test_assembly.py` 时暴露出来的（7 处 subprocess 调用子脚本）。
    text, n = re.subn(r'R / "([A-Za-z_][\w]*\.py)"',
                      lambda m: f'_HERE / "{m.group(1)}"', text)
    if n:
        log.append(f'R / "<脚本>.py" → _HERE / "<脚本>.py"  ×{n}')
    for pat, rep in SUBS:
        # 用 lambda 做替换，避免把 rep 里的反斜杠当转义序列处理
        text, n = pat.subn(lambda _m, _r=rep: _r, text)
        if n:
            log.append(f"{pat.pattern} → {rep}  ×{n}")
    if "d:" + r"\Desktop\2026AiC" in text:
        raise RuntimeError(f"{name}: 仍有未替换的本机路径")
    return text, log


def main() -> int:
    PKG.mkdir(exist_ok=True)
    CODE.mkdir(exist_ok=True)
    print(f"目标包: {PKG}\n")

    print("===== 0) 清理构建残留（__pycache__ 不进交付包）=====")
    n_pyc = 0
    for d in sorted(PKG.rglob("__pycache__")):
        n_pyc += len(list(d.glob("*.pyc")))
        shutil.rmtree(d, ignore_errors=True)
    print(f"  已删 __pycache__ 目录，含 {n_pyc} 个 .pyc")

    print("===== 1) 代码（含路径适配）=====")
    for name, src in FILES:
        if not src.exists():
            print(f"  ✗ 缺源文件 {src}")
            return 1
        text = src.read_text(encoding="utf-8")
        text, rule = inject(text)
        text, log = patch(name, text)
        (CODE / name).write_text(text, encoding="utf-8")
        print(f"  ✓ {name:26s} 注入: {rule}" +
              (f" | 替换 {len(log)} 类" if log else " | 无需替换") +
              (f"\n      {log}" if log else ""))

    print("\n===== 1b) 远端 shell 脚本（原样复制，不注入路径头）=====")
    sdir = CODE / "remote_shell"
    sdir.mkdir(exist_ok=True)
    for name, src in SHELL:
        if not src.exists():
            print(f"  ✗ 缺源文件 {src}")
            return 1
        shutil.copy2(src, sdir / name)
        print(f"  ✓ remote_shell/{name:22s} {src.stat().st_size/1024:.1f} KB")

    print("\n===== 2) 中间产物（算法自产）=====")
    for nm in ARTIFACTS:
        src = R / nm
        if not src.exists():
            print(f"  ✗ 缺 {nm}")
            return 1
        shutil.copy2(src, PKG / nm)
        print(f"  ✓ {nm}  {src.stat().st_size/1024:.1f} KB")

    print("\n===== 2b) 数据集索引（仅 queries.json 小文件）=====")
    # 只为让格式校验器能离线自校验；图像与模型权重属大文件，按赛方要求不入包。
    # queries.json 的内容本就逐条内嵌在结果文件里（四字段回显），故无额外披露。
    q_src = (R / "复赛数据集-基于大模型的多模态视觉理解与推理"
             / "queries" / "queries.json")
    q_dst = (PKG / "复赛数据集-基于大模型的多模态视觉理解与推理"
             / "queries" / "queries.json")
    q_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(q_src, q_dst)
    print(f"  ✓ queries/queries.json  {q_src.stat().st_size/1024:.1f} KB")

    print("\n===== 3) 规则层复跑基线件 =====")
    for rel, dst in BASELINES:
        src = RESULT_DIR / rel
        out = PKG / "submission_zips" / "semifinal" / dst
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        print(f"  ✓ {dst}  {src.stat().st_size/1024:.1f} KB")
    shutil.copy2(RESULT_DIR / "SHA256.txt",
                 PKG / "submission_zips" / "semifinal" / "SHA256.txt")
    print("  ✓ SHA256.txt（提交件谱系）")

    print("\n===== 4) 排行榜结果文件 =====")
    (PKG / "result").mkdir(exist_ok=True)
    for ext in (".zip", ".json"):
        src = RESULT_BEST / f"{RESULT_STEM}{ext}"
        shutil.copy2(src, PKG / "result" / f"{RESULT_STEM}{ext}")
        print(f"  ✓ result/{RESULT_STEM}{ext}  {src.stat().st_size/1024:.1f} KB")
    print(f"    （= 线上最高分件 {RESULT_SCORE}）")

    print("\n===== 4b) 训练与评估脚手架（未采用，仅为环节完整）=====")
    tdir = CODE / "training_未采用"
    tdir.mkdir(exist_ok=True)
    # 与主管线走**同一套**注入+替换，保证包内所有代码路径都可移植
    for nm in TRAINING:
        text = (R / nm).read_text(encoding="utf-8")
        text, rule = inject(text)
        text, log = patch(nm, text)
        (tdir / nm).write_text(text, encoding="utf-8")
        print(f"  ✓ training_未采用/{nm:26s} {rule}"
              + (f" | 替换 {len(log)} 类" if log else " | 无需替换"))

    print("\n===== 5) 环境实录（模型文件与环境）=====")
    # 实测环境的一次性记录：驱动/CUDA/包版本/模型分片哈希/服务命令行。
    # 事后重建只能得到"另一次解析结果"，不等于当时跑出产物的环境，故必须随包提交。
    shutil.copy2(R / "_env_record.txt", PKG / "env_record.txt")
    print(f"  ✓ env_record.txt  "
          f"{(R / '_env_record.txt').stat().st_size / 1024:.1f} KB")

    print(f"\n打包完成：{PKG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
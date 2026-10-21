# -*- coding: utf-8 -*-
"""阈值校准：用真实 38 条探针数据跑 _merge_locate_gd.py 的融合逻辑。

关键检查：探针中"Locate 明显更优"（人工判定红色胜出）的样本
是否被 R1/R2 回退到基线；若被回退则阈值过激，需放宽。
"""
import json
import re
from pathlib import Path

R = Path(__file__).resolve().parent
PROBE = R / "probe_result.json"              # 真实 Locate 输出（boxes 格式）
BASE = R / "result_gd_3modal_fullseq.json"   # 已锁定基线
QUERIES = R / "初赛数据集-基于大模型的多模态视觉理解与推理" / "queries" / "queries.json"

AREA_LIMIT = 0.35
IoU_LOW = 0.2

# 人工判定"Locate 更优"的探针样本（指数标）
KNOWN_LOCATE_BETTER = {
    "001038_003",   # 最左侧轿车：Locate 在左，基线在右
    "000023_002",   # second stone pier：Locate 第2石墩
    "000023_003",   # third stone pier
    "000212_001",   # 霓虹 M 字母
    "001038_009",   # 左车牌小螺丝（Locate 更细）
    "000023_001",   # first stone pier
    "000023_005",   # 栅栏后石墩
}
# 人工判定"基线更优/接近"的样本
KNOWN_BASE_OK = {
    "000002_002",   # 两把白色雨伞
    "000002_003",   # 格纹衬衫的人
}


def iou(a, b):
    if not a or not b:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2 - x1), max(0., y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 1e-9 else 0.0


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def main():
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    base = json.loads(BASE.read_text(encoding="utf-8"))
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))

    print(f"探针 {len(probe)} 条，基线 {len(base)} 条\n")
    print(f"{'qid':<12} {'loc_hit':<8} {'loc_area':<9} {'base_area':<10} "
          f"{'IoU':<6} {'决策':<8} 说明")
    print("-" * 88)

    n_loc = n_back = 0
    bad_cases = []
    for k in sorted(probe):
        boxes = probe[k].get("boxes") or []
        lb = boxes[0] if boxes else None
        bb = base[k]["bbox"]
        q = queries[k]["query"]
        reason = ""
        use_locate = True
        if lb is None:
            use_locate = False
            reason = "R0 无框"
        elif area(lb) > AREA_LIMIT:
            use_locate = False
            reason = f"R1 面积{area(lb):.2f}>0.35"
        elif iou(lb, bb) < IoU_LOW:
            al, ab = area(lb), area(bb)
            if al > 1.5 * ab:
                use_locate = False
                reason = f"R2 放大漂移 IoU={iou(lb,bb):.2f} 面积{al:.3f}/{ab:.3f}"
            else:
                reason = f"IoU={iou(lb,bb):.2f} 缩小/接近，保留"
        else:
            reason = f"IoU={iou(lb,bb):.2f}"

        n_loc += use_locate
        n_back += (not use_locate)
        mark = ""
        if k in KNOWN_LOCATE_BETTER:
            mark = "<== ★Locate更优"
        elif k in KNOWN_BASE_OK:
            mark = "<== ✓基线OK"
        if lb is None:
            print(f"{k:<12} {'NO':<8} {'—':<9} {area(bb):<10.4f} {'—':<6} "
                  f"{'回退':<8} {reason} {mark}")
        else:
            print(f"{k:<12} {'YES':<8} {area(lb):<9.4f} {area(bb):<10.4f} "
                  f"{iou(lb,bb):<6.3f} {'Locate' if use_locate else '回退':<8} "
                  f"{reason} {mark}")
        if use_locate is False and k in KNOWN_LOCATE_BETTER:
            bad_cases.append(k)

    print("-" * 88)
    print(f"采用 Locate {n_loc} / 回退 {n_back} / 共 {len(probe)}")
    print("\n===== 阈值健康性判定 =====")
    if bad_cases:
        print(f"[BAD] Locate 更优却被回退 {len(bad_cases)} 条: {bad_cases}")
        print("→ 阈值过激，需放宽 R1/R2 或按类别豁免 ord/spat")
    else:
        print("[OK] 所有人工判定 'Locate更优' 样本均保留（未被 R1/R2 回退）")
        print("→ 阈值方向正确，可复用同规则到全量")


if __name__ == "__main__":
    main()
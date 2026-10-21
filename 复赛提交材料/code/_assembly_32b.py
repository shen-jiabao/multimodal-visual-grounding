# -*- coding: utf-8 -*-
"""32B 非序数替换的装配器（离线）。

来源与口径
- 基线：`result_semi_enum_trunc`（线上 0.7598）= 当前最优提交件
- 新证据：32B 在 4587 条非序数上的直接框（`qwen32b_direct.json`）
- 序数层 1103 条**原样保留**——这是隔离变量的关键：本次提交只改非序数，
  得分一动就能干净归因。读数口径（★据实）：
      `得分 × 5690 − 675 = 本次非序数桶的正确数`
  它混含「32B 生效的部分」与「保留基线的部分」（对照：现行该桶 3648 条），
  **不能直接当作 32B 自身的准确率**——要分离须把保留基线的那些条再归零测一次。

覆盖率优先（实测必需）
    32B 对约 13% 的条目输出 `[]`（模型认为图中无该物体，raw 就是空数组，不是格式问题）。
    这类条目**必须保留基线框**——留空必然错，而基线框至少还有猜中的机会。
    这与项目既有的"所有无框条目必须由确定来源补齐"是同一条纪律。
    保留基线共四类（见主循环，**互斥分类**并带总和断言）：无框 / 框退化逆序 /
    面积>0.35 / 面积过小。其中"面积>0.35"单列，因为基线本身就有 169 条大框，
    这类拒绝可能是损失，需在真实产物上看规模再决定是否放宽 MAX_AREA。

用法
    python -B _assembly_32b.py --direct <32b 产物.json> [--zip]
"""
from __future__ import annotations
# ===== 提交包路径适配（打包时自动注入；算法逻辑未改动）=====
import os as _os
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
ROOT = _Path(_os.environ.get("AIC_ROOT") or _HERE.parent)
ROOT_S = str(ROOT)
# ===== 注入结束 =====

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

R = Path(ROOT_S)
sys.path.insert(0, str(R))

import _fuse_qwen as F  # noqa: E402
import _sub_path as SP  # noqa: E402

SEMI = R / "复赛数据集-基于大模型的多模态视觉理解与推理"
CORES = R / "_ordinal_cores_r2.json"
BASE = "result_semi_enum_trunc"
N_ORD_CORRECT = 675        # 归零法实测：现有序数层的正确条数（用于反推 32B 成绩）


def load(direct_path: Path, base_name: str | None = None):
    """`base_name` 缺省即模块常量（原行为），加参数是为了把本装配器复用到更高基线
    （如 0.7766 件）之上——**默认不变**，故既有调用与夹具回归不受影响。"""
    q = json.loads((SEMI / "queries" / "queries.json").read_text(encoding="utf-8"))
    cores = set(json.loads(CORES.read_text(encoding="utf-8")))
    base = json.loads(SP.find(base_name or BASE).read_text(encoding="utf-8"))
    direct = json.loads(direct_path.read_text(encoding="utf-8"))
    return q, cores, base, direct


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", required=True)
    ap.add_argument("--stem", default="result_semi_n32_ordbase")
    ap.add_argument("--base", default=BASE,
                    help="基线提交件名；缺省即 result_semi_enum_trunc（原行为）")
    ap.add_argument("--zip", action="store_true")
    # 读数口径的三个锚点。缺省即 0.7598 基线（原行为，夹具回归不受影响）；
    # 换更高基线时必须一并改这三个数，否则打印出来的读数公式是错的。
    ap.add_argument("--base-score", type=float, default=0.7598,
                    help="基线线上得分；缺省 0.7598")
    ap.add_argument("--ord-correct", type=int, default=N_ORD_CORRECT,
                    help=f"基线序数层正确条数；缺省 {N_ORD_CORRECT}")
    ap.add_argument("--base-correct", type=int, default=3648,
                    help="基线非序数桶正确条数（作对照）；缺省 3648")
    a = ap.parse_args()

    q, cores, base, direct = load(Path(a.direct), a.base)
    nonord = [k for k in base if k not in cores]
    print(f"基线 {a.base}（{len(base)} 条）| 序数 {len(cores)} | 非序数 {len(nonord)}")
    print(f"32B 产物 {len(direct)} 条")

    st = Counter()
    changes = []
    big_rejected = []          # 仅因面积>MAX_AREA 被拒的条目（单独看，防漏损）
    out = {k: dict(v) for k, v in base.items()}
    n_base_big = 0
    for k in nonord:
        if F.area(base[k]["bbox"]) > F.MAX_AREA:
            n_base_big += 1
        rec = direct.get(k) or {}
        b = rec.get("bbox")
        # ★以下分类**互斥**——每条只落一个桶。曾因"有合法框"是超集而总和对不上
        #   （4616 > 4587），会让人误判某类数量。末尾有总和断言兜底。
        if not b:
            st["1 无框 → 保留基线"] += 1
            continue
        nb = F.norm(b)
        if not F.ordered(nb):
            st["2 框退化/逆序 → 保留基线"] += 1
            continue
        if F.area(nb) > F.MAX_AREA:
            # 单列：这类拒绝可能是**损失**（大目标），需在真实产物上看规模再决定是否放宽
            st["3 面积>0.35 → 保留基线"] += 1
            big_rejected.append((k, round(F.area(nb), 3),
                                 round(F.area(base[k]["bbox"]), 3),
                                 F.iou(nb, base[k]["bbox"])))
            continue
        if F.area(nb) < F.MIN_AREA:
            st["4 面积过小 → 保留基线"] += 1
            continue
        if F.iou(nb, base[k]["bbox"]) >= F.IOU_KEEP:
            st["5 与基线一致(IoU≥0.5) → 不动"] += 1
            continue
        st["6 ★改动"] += 1
        changes.append((k, base[k]["bbox"], nb, F.iou(nb, base[k]["bbox"])))
        out[k]["bbox"] = nb

    for k, v in sorted(st.items()):
        print(f"  {k:32s} {v}")
    tot = sum(st.values())
    assert tot == len(nonord), f"分类总和 {tot} ≠ 非序数 {len(nonord)}（分类不互斥）"
    print(f"  分类合计 {tot} = 非序数 {len(nonord)} ✓（互斥口径自检通过）")
    print(f"\n  参考：基线中面积>0.35 的条目 {n_base_big} 条（这些条目上若 32B 也给大框，"
          f"会被第 3 类拒掉）")
    if big_rejected:
        loss = [_t for _t in big_rejected if _t[2] <= F.MAX_AREA]
        print(f"  ★仅因面积超限被拒 {len(big_rejected)} 条："
              f"其中基线也>0.35 的 {len(big_rejected) - len(loss)} 条"
              f"（32B 与基线同为大框，不动风险低）；"
              f"★基线≤0.35 的 {len(loss)} 条 —— 名义上是**潜在损失**"
              f"（32B 给了大框、基线是小框）")
        # 全量列出"潜在损失"（条数本就不多），**不抽样**——是否放宽 MAX_AREA 全看这批
        # 的成色：看 IoU（与基线）。若 IoU 高说明两框其实一致，守卫并未丢信息，
        # 放宽只会把大框放进来、得不偿失；只有 IoU 低才是真有内容被吃掉。
        if loss:
            print(f"    潜在损失全量（qid, 32B面积, 基线面积, 与基线IoU）:")
            for _k, _a, ab, io in loss:
                print(f"      {_k}  32B={_a:.3f}  基线={ab:.3f}  IoU={io:.3f}")
        rest = [_t for _t in big_rejected if _t[2] > F.MAX_AREA]
        print(f"    基线也>0.35 的样例: {[(r[0], r[1], r[2]) for r in rest[:4]]}")

    # ---- 产物健康度：供中途与跑完后同一口径查看，不必等跑完 ----
    boxed = [k for k in nonord if (direct.get(k) or {}).get("bbox")]
    ars = sorted(F.area(F.norm(direct[k]["bbox"])) for k in boxed)
    if ars:
        qs = lambda p: ars[min(int(len(ars) * p), len(ars) - 1)]
        # 中途读部分产物时，"桶 1" 会同时含「真的输出 []」与「尚未跑到」两种，
        # 必须分开显示，否则会把未跑到的当成大量无框而误读。
        no_box = [k for k in nonord
                  if k in direct and not (direct.get(k) or {}).get("bbox")]
        absent = [k for k in nonord if k not in direct]
        agree = st["5 与基线一致(IoU≥0.5) → 不动"]
        print(f"\n  健康度（统一口径，可中途看）：已出 {len(direct)}/{len(nonord)} 条")
        print(f"    有框 {len(boxed)}（占已出 {len(boxed)/max(len(direct),1):.1%}）｜"
              f"已跑过但输出 [] {len(no_box)}（占已出 "
              f"{len(no_box)/max(len(direct),1):.1%}）｜尚未跑到 {len(absent)}")
        print(f"    框面积 p10 {qs(.1):.4f} / p50 {qs(.5):.4f} / p90 {qs(.9):.4f} "
              f"/ max {ars[-1]:.4f}")
        print(f"    面积>0.35 的 {sum(1 for a in ars if a > F.MAX_AREA)} 条｜"
              f"<1e-4 的 {sum(1 for a in ars if a < 1e-4)} 条｜"
              f"整幅(≥0.99) 的 {sum(1 for a in ars if a >= 0.99)} 条")
        print(f"    与基线一致率（IoU≥0.5）: {agree}/{len(boxed)} = "
              f"{agree/max(len(boxed),1):.1%}")
    n = len(changes)
    print(f"\n★ 实际改动 {n} 条 | 若净成功率与同族中位(25%)相同 ⇒ 预期 "
          f"{n*0.25:.0f} 条 → ≈ {(a.base_score + n*0.25/5690):.4f}")
    print(f"  读数口径（★据实）：得分×5690 − {a.ord_correct} = 本次**非序数桶**的正确数"
          f"，它混合了新模型生效的部分与保留基线的部分"
          f"（对照：基线该桶 {a.base_correct} 条）")
    print(f"  即：该读数**不能直接当作新模型自身的准确率**——要分离需再测一次"
          f"（把保留基线的 {st['1 无框 → 保留基线'] + st['2 框退化/逆序 → 保留基线'] + st['3 面积>0.35 → 保留基线'] + st['4 面积过小 → 保留基线']} 条归零）")
    for k, o, nb, io in changes[:8]:
        print(f"    {k} IoU(与基线)={io:.3f}  {q[k]['query'][:52]}")

    # 硬约束自检（与提交校验器同口径）
    assert len(out) == 5690, len(out)
    assert set(out) == set(q), "键集与 queries 不一致"
    # ★序数层逐位不变：本次提交只改非序数，这是隔离变量的前提。若这条不成立，
    #   读数口径里的 `675`（序数层实测正确数）就不再是同一个东西，整件读数不可解释。
    #   故把它做成**硬断言**——构造保证不等于已核验。
    ord_same = sum(1 for k in cores if out[k] == base[k])
    assert ord_same == len(cores), f"序数层被动过 {len(cores) - ord_same} 条"
    print(f"自检：序数层 {ord_same}/{len(cores)} 条与基线逐位相同 ✓")
    bad = [k for k in q if any(out[k].get(f) != q[k][f]
                              for f in ("visible", "infrared", "depth", "query"))]
    assert not bad, f"字段回显异常 {len(bad)} 条"
    leg = [k for k, v in out.items()
           if not (len(v["bbox"]) == 4 and 0 <= v["bbox"][0] < v["bbox"][2] <= 1
                   and 0 <= v["bbox"][1] < v["bbox"][3] <= 1)]
    assert not leg, f"bbox 非法 {len(leg)} 条"
    print("\n自检：条数/键集/字段回显/bbox 合法性 —— 全部通过")

    # ---- 单变量隔离自检（D 路线必需：产物只覆盖作用范围，其余必须逐位保留）----
    ord_moved = [k for k in cores if out[k] != base[k]]
    assert not ord_moved, f"序数层被动过 {len(ord_moved)} 条"
    absent = [k for k in nonord if k not in direct]
    absent_moved = [k for k in absent if out[k] != base[k]]
    assert not absent_moved, f"未提供证据的条目被动过 {len(absent_moved)} 条"
    print(f"单变量隔离自检：序数层 {len(cores)} 条逐位不动 ✓｜"
          f"未提供证据的非序数 {len(absent)} 条逐位不动 ✓")

    if a.zip:
        # 新产物一律落 `未提交/`（已交件按得分归入得分子目录；未交件不属于任何得分目录）。
        # 仍走 `_sub_path` 解析，保证"同一提交件不出现两份"这条约定不被绕过。
        jp, zp = SP.out_path(a.stem, ".json"), SP.out_path(a.stem, ".zip")
        if jp.parent == SP.SUB:
            jp = SP.SUB / "未提交" / jp.name
        if zp.parent == SP.SUB:
            zp = SP.SUB / "未提交" / zp.name
        # ★两个输出都要确保父目录存在：包内首次运行时 `未提交/` 可能尚未创建，
        #   否则写 zip 会 FileNotFoundError（工作区里该目录恰好已存在，故长期未暴露）。
        jp.parent.mkdir(parents=True, exist_ok=True)
        zp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                      encoding="utf-8")
        payload = json.dumps(out, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
            zi = zipfile.ZipInfo(f"{a.stem}.json", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, payload)
        print(f"\n已写 {jp}")
        print(f"     {zp}\n     SHA256 "
              f"{hashlib.sha256(zp.read_bytes()).hexdigest().upper()}")
        r = subprocess.run([sys.executable, "-B", str(_HERE / "_validate_semi.py"),
                            str(zp)], capture_output=True, text=True)
        print("  " + "\n  ".join(ln for ln in r.stdout.splitlines()
                                  if ln.strip().startswith(("[", "✓", "✗"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
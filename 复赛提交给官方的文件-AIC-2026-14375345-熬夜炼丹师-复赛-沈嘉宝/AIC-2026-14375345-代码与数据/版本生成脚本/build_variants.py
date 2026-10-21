# build_variants.py — 由槽位 JSON 生成 core/spatial/ordinal/original 查询变体（纯 CPU，不烧 GPU）
# 输入：{qid: {"query": str, "slots": {...}}}
# 输出：{qid: {"original","core","spatial","ordinal","meta"}}
# 变体语义（与第 3、4 层配套）：
#   core     -> 最短名词短语，喂 GroundingDINO（BERT 文本塔吃短句更准）
#   spatial  -> 带空间关系短语，喂 LocateAnything / VLX（VLM 能读关系）
#   ordinal  -> 序数查询去掉序数的检测短语（拿全部同类候选，由规则取第 N 个）
#   original -> 原句，对照轨
import json
import sys

COLOR = {"white", "black", "grey", "gray", "blue", "red", "yellow", "green",
         "orange", "pink", "brown", "silver", "gold", "purple"}

REL_READ = {"in_front_of": "in front of", "left_of": "left of",
            "right_of": "right of", "next_to": "next to", "on": "on",
            "under": "under", "behind": "behind", "above": "above",
            "below": "below"}


# 大区域误判守卫：head 是"物体名词"（含 sign/bollard/lamp/pole/arm 等后缀）时，
# 即使槽位 is_large_region 误置 true，也强制转回物体（避免 road sign 等被当作大区域）。
OBJECT_NOUN_MARK = ("sign", "bollard", "lamp", "post", "pole", "arm", "chair",
                    "car", "vehicle", "bench", "panel", "person",
                    "deer", "monkey", "bear", "pelican", "paddle", "cushion",
                    "signboard")


def _fix_large_region(large: bool, head: str) -> bool:
    """head 命中物体名词时视为物体而非大区域。"""
    h = head.lower()
    if any(tok in h for tok in OBJECT_NOUN_MARK):
        return False
    return large


def build_variants(qid: str, query: str, slots: dict) -> dict:
    """根据槽位生成各查询变体。"""
    head = str(slots.get("head", "")).strip()
    attrs = [a for a in slots.get("attrs", []) if isinstance(a, str)]
    rel = slots.get("relation") or {}
    ordinal = slots.get("ordinal")
    axis = slots.get("axis")
    count = slots.get("count")
    large = bool(slots.get("is_large_region", False))
    large = _fix_large_region(large, head)

    # core：head + 首个颜色属性；若 head 已含该颜色词则不重复前拼
    color_attr = next((a.lower() for a in attrs if a.lower() in COLOR), None)
    dup = (color_attr and
           (head.lower().startswith(color_attr + " ") or head.lower() == color_attr))
    core = f"{color_attr} {head}".strip() if color_attr and not dup else head

    # spatial：有实体关系时拼 "head type anchor"
    if isinstance(rel, dict) and rel.get("type") and rel.get("anchor"):
        t = REL_READ.get(str(rel["type"]), str(rel["type"]))
        spatial = f"{head} {t} {rel['anchor']}".strip()
    else:
        spatial = core

    # ordinal：序数查询保留 head 作为全量检测短语，否则置空
    ordinal_variant = head if ordinal is not None else None

    return {
        "original": query,
        "core": core,
        "spatial": spatial,
        "ordinal": ordinal_variant,
        "meta": {
            "ordinal": ordinal,
            "axis": axis,
            "count": count,
            "is_large_region": large,
            "relation": rel if isinstance(rel, dict) and rel else None,
        },
    }


def main() -> int:
    in_path = sys.argv[1] if len(sys.argv) > 1 else "_slot30.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "variants_30.json"
    data = json.load(open(in_path, encoding="utf-8"))

    result = {qid: build_variants(qid, d["query"], d["slots"])
              for qid, d in data.items()}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    # 打印可读对照表
    for qid, v in result.items():
        m = v["meta"]
        tag = []
        if m["ordinal"] is not None:
            tag.append(f"ord={m['ordinal']} axis={m['axis']}")
        if m["count"] is not None:
            tag.append(f"count={m['count']}")
        if m["relation"]:
            tag.append(f"rel={m['relation']['type']}:{m['relation']['anchor']}")
        if m["is_large_region"]:
            tag.append("LARGE")
        print(f"--- {qid}  [{', '.join(tag) if tag else 'plain'}]")
        print(f"  Q     : {v['original']}")
        if v["ordinal"]:
            print(f"  ordinal: {v['ordinal']}")
        print(f"  core  : {v['core']}")
        if v["spatial"] != v["core"]:
            print(f"  spatial: {v['spatial']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# parse_slots.py — 分层采样 50 条，Qwen2.5-3B 解析槽位并输出对照表
import json
import random
import re
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/models/Qwen2.5-3B-Instruct"
IN_Q = "/root/dataset/初赛数据集-基于大模型的多模态视觉理解与推理/queries/queries.json"

SYS = """Extract structured slots from an English visual-grounding query.
Return ONLY a JSON object with these VALID keys and types:
  "head": string          # the exact MINIMAL target object noun phrase = the actual thing being asked for
  "attrs": [string]       # color/material/size/shape adjectives, only those describing the head
  "relation": null-or-obj # non-null ONLY when an explicit spatial preposition links the head to a SEPARATE concrete object
  "ordinal": null | int   # int ONLY for first/second/third/.../last; else null. NEVER a word.
  "count": null | "all" | int   # only a stated NUMBER of separate items
  "axis": null | "left_to_right" | "right_to_left" | "top_to_bottom" | "bottom_to_top"
  "is_large_region": bool

DECIDE BY EXACT BRANCH, in this order (first matching branch wins):
A) Query contains a literal ordinal word (first/second/third/.../last):
   IF-THEN hard rule: "ordinal"=<int>; "relation"=null ALWAYS (never emit a relation); "axis" from direction phrase.
   -> "head"=the target noun WITHOUT the ordinal word.

B) Query contains "closest"/"nearest"/"farthest"/"furthest"/"far"/"near" (relative distance, NOT an ordinal):
   -> "ordinal"=null, "relation"=null, "axis"=null. Keep head/attrs only.

C) Query places the head relative to a SEPARATE concrete object via in front of / behind / next to / below / above / under / on / left of / right of / beside / between:
   -> "relation"={"type": <preposition>, "anchor": <the SEPARATE concrete object noun>}.
   -> "ordinal"=null.

D) HARD RULE FOR EDGE/POSITION PHRASES -> relation MUST be null:
   These position the head relative to the IMAGE, not to a separate object -> NO relation, only axis:
   "on the far right / far left of the image", "right side / left side of the scene/image/picture",
   "in the right/left group", "upper / lower part", "left edge / right edge", "top / bottom of the image",
   "right hand of the person" (possession, not spatial left-of/right-of),
   "inserted into the ground", "on the ground", "on the scene".
   -> "relation"=null ; set "axis"=left_to_right / right_to_left / top_to_bottom / bottom_to_top if a clear side is named.

E) HEAD DIRECTIONALITY (always): the head is the thing ASKED FOR, not what it is near/on/looking-at.
   - "the person/people looking at the elephant" -> head=person/people (elephant is only an object the person watches -> do NOT make it the head).
   - "the person sitting on the paddleboard" -> head=person (paddleboard is the support, not the target).
   - A postverbal "on/in front of/beside..." that names a container/surface (ground/path/room/wall/fence/car) is an anchor only if it is the actual separate object the head is placed against; otherwise drop it.

ADDITIONAL hard rules:
- "all the X" -> "count"="all"; "two/four/..." separate items -> "count"=int. NEVER derive count from a compound noun (three-wheeler/victoria/pullman is ONE vehicle -> "count"=null). NEVER mark "all" for mass nouns like lake water/path/ground.
- Never output a word for "ordinal": "far", "closest", "nearest", "left", "right" are never ordinals -> "ordinal"=null unless a literal first/second/.../last token exists.
- "head" is MINIMAL: drop the ordinal word and "all"/numbers and trailing relational phrases. Examples: "white fonts" not the sign; "right hand" not the person; "three-wheeler" not "first three-wheeler"; person not "person wearing pink pants" (keep the noun, drop the phrase).
- "is_large_region": ALWAYS true when the head itself is a big ground/surface region: land, water, lake, sea, sky, fence, path, walkway, road, grass, field, ground, pavement, soil, wall, or when the query says right half/left half/entire/whole/top of the image/bottom of the image. mass nouns (water/path/ground/wall/land/lake) hence get is_large_region=true even without the word "large".
- Never invent colors/materials/details not stated."""

ORD = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)\b", re.I)
SPAT = re.compile(r"leftmost|rightmost|topmost|bottommost|nearest|closest|farthest|furthest|"
                  r"\bleft\b|\bright\b|\btop\b|\bbottom\b|behind|in front|above|below|"
                  r"beside|next to|between|foreground|background|upper|lower", re.I)
COUNT = re.compile(r"\b(all the |two|three|four|five|six|seven|eight|nine|ten|both|a pair of|several)\b", re.I)
LARGE = re.compile(r"\b(land|water|lake|sky|fence|path|walkway|grass|field|ground|pavement|soil|wall|sea)\b|"
                   r"(right half|left half|entire|whole|large area|top of the image|bottom of the image)", re.I)


# ---- 确定性守卫：把小模型不稳的硬规则用正则清零 ----
ORD_WORDS = r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last"
CARD_MANY = r"\b(?:two|three|four|five|six)\s+[a-z]+"
SURFACE = r"ground|scene|image|picture|room|path|sign|edge|group|background|foreground|ceiling|\bleft\b|\bright\b|\btop\b|\bbottom\b|\bfar\b"
BIG_HEAD = r"\b(land|water|lake|sea|sky|fence|path|walkway|road|grass|field|ground|pavement|soil|wall)\b"
BIG_Q = r"(right half|left half|entire|whole|top of the image|bottom of the image)"
VALID_REL = ("left_of", "right_of", "above", "below", "in_front_of",
             "behind", "next_to", "on", "under")


def _norm_rel_type(t: str) -> str | None:
    """把关系类型归一成合法集合；不合法返回 None（置空）。"""
    t = (t or "").strip().lower().replace(" ", "_")
    if t in VALID_REL:
        return t
    return None


def normalize(d: dict, q: str) -> dict:
    """确定性守卫，固定已知失败点。d 为 LLM 草稿，q 为原句。"""
    ql = q.lower()

    # 1) ordinal：仅当句中出现字面序数且值可转 int 才保留
    if not re.search(r"\b(?:%s)\b" % ORD_WORDS, ql):
        d["ordinal"] = None
    elif isinstance(d.get("ordinal"), int):
        pass  # 保留
    else:
        # 值非法（字符串如 far/closest）但句中有序数 -> 尝试重取
        m = re.search(r"\b(?:%s)\b" % ORD_WORDS, ql)
        d["ordinal"] = (list(m.group(0) for _ in [0]) and
                        {"first":1,"second":2,"third":3,"fourth":4,"fifth":5,
                         "sixth":6,"seventh":7,"eighth":8,"ninth":9,"tenth":10,
                         "last":None}[m.group(0).lower()])

    # 2) count：仅当 "all the" 或显式独立数量词才保留，否则清零
    if re.search(r"\ball the\b", ql):
        d["count"] = "all"
    elif isinstance(d.get("count"), int) and re.search(CARD_MANY, ql):
        pass  # 保留显式数量
    else:
        d["count"] = None

    # 3) Rule A：ordinal 非空 -> relation 恒 null
    if d.get("ordinal") is not None:
        d["relation"] = None

    # 4) relation：类型合法且锚点非表面词才保留
    rel = d.get("relation") or {}
    if isinstance(rel, dict) and rel.get("anchor"):
        t = _norm_rel_type(rel.get("type", ""))
        anchor = str(rel.get("anchor", "")).lower()
        if t is None or re.search(SURFACE, anchor):
            d["relation"] = None
        else:
            rel["type"] = t
            d["relation"] = rel

    # 5) head：剥离前导冠词/序数（机械最小化）
    head = str(d.get("head", "")).strip()
    head = re.sub(
        r"^(?:(?:the|a|an)\s+)+(?:%s)\s+" % ORD_WORDS, "", head, count=1, flags=re.I)
    head = re.sub(r"^(?:the|a|an)\s+", "", head, count=1, flags=re.I)
    d["head"] = head

    # 6) is_large_region：head 或原句命中大区域 -> true
    if re.search(BIG_HEAD, head.lower()) or re.search(BIG_Q, ql):
        d["is_large_region"] = True
    else:
        d.setdefault("is_large_region", False)

    return d


def parse_one(model, tok, q):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=256, do_sample=False)
    resp = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    m = re.search(r"\{.*\}", resp, re.S)
    return json.loads(m.group(0)) if m else {"_parse_error": resp}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    seed = 42
    random.seed(seed)

    Q = json.load(open(IN_Q))
    qd = {k: v["query"] for k, v in Q.items()}

    # 分层：显式锚点
    ord_explicit = ["000023_002", "000023_003", "000023_004"]
    spa_explicit = ["000023_006"]
    lrg_explicit = ["000478_005", "015248_003", "016606_001"]

    # 其余从池子里按正则补足
    ord_pool = [k for k, v in qd.items()
                if k not in ord_explicit and ORD.search(v)]
    spa_pool = [k for k, v in qd.items()
                if k not in spa_explicit and (not ORD.search(v)) and SPAT.search(v)
                and not COUNT.search(v)]
    lrg_pool = [k for k, v in qd.items()
                if k not in lrg_explicit and LARGE.search(v) and not ORD.search(v)]

    random.shuffle(ord_pool); random.shuffle(spa_pool); random.shuffle(lrg_pool)

    ord_pick = ord_explicit + ord_pool[:10 - len(ord_explicit)]
    spa_pick = spa_explicit + spa_pool[:10 - len(spa_explicit)]
    lrg_pick = lrg_explicit + lrg_pool[:10 - len(lrg_explicit)]

    used = set(ord_pick) | set(spa_pick) | set(lrg_pick)
    rest = [k for k in qd if k not in used]
    random.shuffle(rest)
    rand_pick = rest[:20]

    keys = ord_pick + spa_pick + lrg_pick + rand_pick
    labels = (["ordinal"]*len(ord_pick) + ["spatial"]*len(spa_pick)
              + ["large/count"]*len(lrg_pick) + ["random"]*len(rand_pick))
    print(f"分层: ordinal {len(ord_pick)}  spatial {len(spa_pick)}  "
          f"large/count {len(lrg_pick)}  random {len(rand_pick)}  总计 {len(keys)}",
          file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype="auto", device_map="cuda", trust_remote_code=True)

    slots = {}
    for k, lab in zip(keys, labels):
        draft = parse_one(model, tok, qd[k])
        slots[k] = {"_cat": lab, **normalize(draft, qd[k])}
        print(f"--- [{lab}] {k}\n  Q: {qd[k]}\n  S: {json.dumps(slots[k], ensure_ascii=False)}",
              flush=True)

    json.dump(slots, open(f"/root/output/query_slots_{n}_strat.json", "w", ) if False
              else open(f"/root/output/query_slots_{n}_strat.json", "w"), indent=1)
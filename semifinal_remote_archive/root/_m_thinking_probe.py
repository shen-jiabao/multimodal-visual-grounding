#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预登记 M §3.3「未知项」的实测：vLLM 是否接受 chat_template_kwargs，
以及 thinking 开/关对**原始输出**的影响。

做法：同一条 query、同一张图，分别以 THINKING=default 与 THINKING=0 各打一次，
打印原始输出长度、是否含 `<think>`、前 200 字符与解析结果。
（gen 内部读的是模块级 THINKING，故运行时改 P.THINKING 即生效。）
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

import _qwen_full_pipeline as P


def main() -> int:
    data = P.DATA
    q = json.load(open(os.path.join(data, "queries", "queries.json"),
                       encoding="utf-8"))
    ks = sorted(q) if len(sys.argv) < 2 else sys.argv[1:]
    k = ks[0]
    vis = Image.open(os.path.join(data, q[k]["visible"])).convert("RGB")
    if P.MODALITIES == "3":
        img_in = [vis,
                  P.load_ir(os.path.join(data, q[k]["infrared"])),
                  P.load_depth(os.path.join(data, q[k]["depth"]))]
        tmpl = P.THREE_IMG_PROMPT
    elif P.MODALITIES == "3m":
        img_in = P.merge3(vis,
                          P.load_ir(os.path.join(data, q[k]["infrared"])),
                          P.load_depth(os.path.join(data, q[k]["depth"])))
        tmpl = P.MERGED_PROMPT
    else:
        img_in = vis
        tmpl = P.FULL_DIRECT_PROMPT
    prompt = tmpl.format(q=q[k]["query"], rules=P.RULES_BLOCK)

    print(f"qid={k}｜模态={P.MODALITIES}｜规则5={P.RULES5}｜"
          f"max_new={P.MAXNEW}｜multi_box={P.MULTIBOX}")
    print(f"Q: {q[k]['query']}")
    gen = P.make_gen_api()

    for think, label in (("default", "默认（模板视为开，effort=xhigh）"),
                         ("0", "显式关闭 enable_thinking=False")):
        P.THINKING = think
        print(f"\n--- THINKING={think}  {label} ---", flush=True)
        try:
            txt = gen(img_in, prompt)
        except Exception as ex:
            print(f"  ★请求失败：{type(ex).__name__}: {str(ex)[:300]}")
            print("  ⇒ 若报『未知字段 / 无法解析』，按预登记 M §8 R1 改起服参数 "
                  "--default-chat-template-kwargs")
            continue
        print(f"  长度 {len(txt)} 字符｜含 '<think'：{'<think' in txt}")
        print(f"  前 200 字符：{txt[:200]!r}")
        bs = P.parse_boxes(txt, P._img_wh(img_in))
        print(f"  parse_boxes → {len(bs)} 个框，首个 {bs[0] if bs else '—'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

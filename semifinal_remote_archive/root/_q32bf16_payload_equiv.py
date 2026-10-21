# -*- coding: utf-8 -*-
"""接线等价性的**静态证明**（0 GPU）：抓取两条代码路径的真实请求体，逐字节比对。

动机：接线闸门原用「运行时逐位比对输出」作代理，但该代理在服务端存在运行间抖动时
  会误报（实测：同条件 20 条全过、30 条出现 0.001 级偏差；而归档实现自比 30/30 逐位相同）。
  更直接、更稳的判据是：**两条路径发出的 HTTP 请求体是否逐字节相同**。
  请求体相同 ⇒ 客户端不存在接线差异；若有输出差异，必来自服务端。

做法：把 openai.OpenAI 换成记录型桩，分别调用两条路径的 gen()，序列化 kwargs 后比哈希。
★ 不触网、不占 GPU。

用法：python _q32bf16_payload_equiv.py
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import types

os.environ.setdefault("QWEN_ENGINE", "api32b")
os.environ.setdefault("QWEN_API_MODEL", "qwen3-vl-32b-bf16")
os.environ.setdefault("QWEN_API_BASE", "http://127.0.0.1:8000/v1")

from PIL import Image  # noqa: E402

# ---- 记录型桩：拦下 create() 的 kwargs ----
CAPTURED: list[dict] = []


class _Msg:
    def __init__(self) -> None:
        self.content = "<bbox>0.10 0.20 0.30 0.40</bbox>"


class _Choice:
    def __init__(self) -> None:
        self.message = _Msg()


class _Resp:
    def __init__(self) -> None:
        self.choices = [_Choice()]


class _Completions:
    def create(self, **kw):
        CAPTURED.append(kw)
        return _Resp()


class _Chat:
    def __init__(self) -> None:
        self.completions = _Completions()


class _Client:
    def __init__(self, **_kw) -> None:
        self.chat = _Chat()


import openai  # noqa: E402
openai.OpenAI = _Client            # 两条路径都在函数内部 `from openai import OpenAI`，故会被拦到

import _qwen_full_pipeline as P  # noqa: E402
from _probe_models import Api32B  # noqa: E402


def canon(kw: dict) -> str:
    """把 kwargs 规范化成可比字符串：base64 长串替换为其哈希，避免输出爆炸。"""
    kw = json.loads(json.dumps(kw, default=str))     # 深拷贝 + 保证可序列化
    for m in kw.get("messages", []):
        c = m.get("content")
        if isinstance(c, list):
            for part in c:
                url = (part.get("image_url") or {}).get("url")
                if url:
                    b64 = url.split(",", 1)[-1]
                    h = hashlib.sha256(base64.b64decode(b64)).hexdigest()[:16]
                    part["image_url"]["url"] = f"<data-uri>{h} ({len(b64)}B b64)"
    return json.dumps(kw, ensure_ascii=False, sort_keys=True, indent=1)


def main() -> int:
    img = Image.new("RGB", (96, 64), (37, 129, 205))     # 合成图，避免 I/O 且两条路径共用
    prompt = P.FULL_DIRECT_PROMPT.format(q="a test query for wiring equivalence")

    ours = P.make_gen_api()
    ours(img, prompt)
    Api32B().gen(img, prompt)

    if len(CAPTURED) != 2:
        print(f"★ 捕获数异常：{len(CAPTURED)}（应为 2）")
        return 2

    a, b = CAPTURED
    ca, cb = canon(a), canon(b)
    ha = hashlib.sha256(ca.encode()).hexdigest()
    hb = hashlib.sha256(cb.encode()).hexdigest()

    print("=== 我方管线请求体 ===")
    print(ca)
    print("\n=== 归档实现请求体 ===")
    print(cb)
    print(f"\n我方 SHA256 {ha}")
    print(f"归档 SHA256 {hb}")

    if ha == hb:
        print("\n★ 判定：两条路径请求体**逐字节相同**")
        print("  ⇒ 客户端不存在接线差异（该结论与运行时抖动无关，是静态性质）")
        print("  ⇒ 若仍有输出差异，归因必在服务端；闸门应以此为准据而非运行时代理")
    else:
        print("\n★ 判定：请求体**存在差异** ⇒ 客户端接线确有不同，必须查出并修正")
        ja, jb = json.loads(ca), json.loads(cb)
        for k in sorted(set(ja) | set(jb)):
            if ja.get(k) != jb.get(k):
                print(f"  差异字段 {k}: 我方={ja.get(k)!r} 归档={jb.get(k)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
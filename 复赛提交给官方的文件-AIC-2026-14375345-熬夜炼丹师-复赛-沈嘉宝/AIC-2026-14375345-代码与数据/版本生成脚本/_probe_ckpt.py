# -*- coding: utf-8 -*-
"""远端探针：核查 result_gd_3modal_full.json 的字节数/JSON 结构/规模。"""
import json
import os

p = "/root/infer/result_gd_3modal_full.json"
print("size:", os.path.getsize(p) if os.path.exists(p) else "MISSING")
if not os.path.exists(p):
    raise SystemExit(0)
with open(p, "rb") as f:
    head = f.read(96)
print("head:", repr(head))
try:
    d = json.load(open(p))
    print("top_type:", type(d).__name__)
    if isinstance(d, dict):
        print("keys_len:", len(d))
        print("sample:", list(d)[:3])
except Exception as ex:  # noqa: BLE001
    print("parse_error:", repr(ex))
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        data = f.read()
    print("last_120:", repr(data[-120:]))
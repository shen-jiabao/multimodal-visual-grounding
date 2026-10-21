# -*- coding: utf-8 -*-
"""小样产物外科修复：剔掉「清单外条目」与「err 条目」，使续跑只补真正的缺口。

为什么需要：
  · 清单外 16 条 —— 9 张图报 `img_open` 时，`except` 分支未按 `only_q` 过滤，
    把作用范围外的条目也写进了产物（该 bug 已在管线中修复）。
  · err 26 条 —— 深度加载器对 .jpg 组抛异常所致（已修复）。
两者都已从源头消除；本脚本只清理这次运行已写下的残留，避免续跑时把它们当既有结果。
"""
import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
p = R / "_qwen32b_d3_sample.json"
sample = set(json.loads((R / "_d_sample300.json").read_text(encoding="utf-8")))
d = json.loads(p.read_text(encoding="utf-8"))

extra = set(d) - sample
errs = {k for k, v in d.items() if v.get("err")}
# ★注意：经实测，`err` 里混着「清单外」条目——它们是同一张图失败时由 `except`
#   分支一并写下的（也带 err=img_open）。所以必须分别统计，否则会误判失败率
#   （曾把 26 当成全在清单内，实际清单内只有 10 条）。
err_in_sample = errs & sample
keep = {k: v for k, v in d.items() if k in sample and k not in errs}

print(f"原产物 {len(d)} 条｜清单外 {len(extra)}（均带 err）"
      f"｜err 合计 {len(errs)}，其中清单内 {len(err_in_sample)}"
      f"｜保留 {len(keep)}")
assert len(keep) == len(sample) - len(err_in_sample), "保留数不符"
p.write_text(json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"已就地修复 → {len(keep)} 条；续跑应补 {len(sample) - len(keep)} 条"
      f"（全为深度加载器 .jpg 组所致，该 bug 已修）")
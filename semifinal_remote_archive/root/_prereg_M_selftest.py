#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预登记 M 的自证测试：改动的正确性由机器验，不靠人眼。

覆盖三件事：
  ① **开关全关时提示词与改造前逐字节相同**（这是"既有各件产物不受影响"的机器证明）
  ② 开关全开时 5 条规则落在正确位置、`{rules}` 槽展开正确
  ③ parse_multibox / _scale4 的解析正确性（含乱序 idx、null、非 JSON 回退）

用法：python3 _prereg_M_selftest.py     （需 pillow；venv_q38 具备）
"""
import importlib
import os
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ★改造前（冻结）的提示词原文，用于 ① 的逐字节比对
ORIG_FULL = (
    "Locate the object described below in this image and output its bounding "
    "box.\nDescription: {q}\n"
    "Output ONLY a JSON array like "
    '[{{"bbox_2d": [x1, y1, x2, y2], "label": "object"}}] '
    "where coordinates are integers normalized to 0-1000 "
    "(x1,y1 = top-left, x2,y2 = bottom-right)."
).format(q="Q", rules="")
ORIG_THREE = (
    "The three images below show the same scene in this fixed order: "
    "(1) visible-light colour photo, (2) thermal infrared (brighter = hotter), "
    "(3) depth map (brighter = farther, black = invalid depth).\n"
    "Locate the object described below and output its bounding box in the "
    "visible-light image (image 1).\nDescription: {q}\n"
    "Output ONLY a JSON array like "
    '[{{"bbox_2d": [x1, y1, x2, y2], "label": "object"}}] '
    "where coordinates are integers normalized to 0-1000 "
    "(x1,y1 = top-left, x2,y2 = bottom-right)."
).format(q="Q", rules="")

fails = []
nchk = 0


def chk(name, cond, extra=""):
    global nchk
    nchk += 1
    print(("  ✓ " if cond else "  ✗ ") + name + ("" if cond else "   " + extra))
    if not cond:
        fails.append(name)


# ---------------- ① 开关全关：逐字节相同 ----------------
for k in ("QWEN_RULES5", "QWEN_MULTIBOX", "QWEN_THINKING", "QWEN_MAXNEW",
          "QWEN_MODALITIES", "QWEN_ENGINE"):
    os.environ.pop(k, None)
os.environ["QWEN_ENGINE"] = "api32b"
P = importlib.import_module("_qwen_full_pipeline")

print("=== ① 开关全关（= 改造前行为）===")
chk("RULES_BLOCK 为空串", P.RULES_BLOCK == "", repr(P.RULES_BLOCK))
chk("MAXNEW=256", P.MAXNEW == 256, str(P.MAXNEW))
chk("THINKING=default", P.THINKING == "default", P.THINKING)
got_f = P.FULL_DIRECT_PROMPT.format(q="Q", rules=P.RULES_BLOCK)
got_t = P.THREE_IMG_PROMPT.format(q="Q", rules=P.RULES_BLOCK)
chk("FULL_DIRECT_PROMPT 逐字节相同", got_f == ORIG_FULL)
chk("THREE_IMG_PROMPT 逐字节相同", got_t == ORIG_THREE)
if got_f != ORIG_FULL:
    print("     got:", repr(got_f))
    print("     exp:", repr(ORIG_FULL))
chk("既有断言仍成立", P.THREE_IMG_PROMPT.split("{q}")[1]
    == P.FULL_DIRECT_PROMPT.split("{q}")[1])

# ---------------- ③ 解析（在默认配置下即可测） ----------------
print("=== ③ parse_multibox / _scale4 ===")
ok = '[{"idx":1,"bbox_2d":[10,20,30,40]},{"idx":2,"bbox_2d":null},' \
     '{"idx":3,"bbox_2d":[50,60,70,80]}]'
r = P.parse_multibox(ok, 3)
chk("正常 JSON：条数 3", len(r) == 3, str(len(r)))
chk("正常 JSON：框值正确",
    r[0] == [0.01, 0.02, 0.03, 0.04] and r[1] is None
    and r[2] == [0.05, 0.06, 0.07, 0.08], str(r))

rev = '[{"idx":3,"bbox_2d":[50,60,70,80]},{"idx":1,"bbox_2d":[10,20,30,40]},' \
      '{"idx":2,"bbox_2d":[90,90,90,90]}]'
r2 = P.parse_multibox(rev, 3)
chk("乱序 idx：按 idx 定位（不是按数组顺序）",
    r2[0] == [0.01, 0.02, 0.03, 0.04] and r2[2] == [0.05, 0.06, 0.07, 0.08],
    str(r2))

bad = 'blah {"idx": 2, "bbox_2d": [100, 200, 300, 400]} trailing noise'
r3 = P.parse_multibox(bad, 3)
chk("非 JSON：正则回退取到第 2 条",
    r3[0] is None and r3[1] == [0.1, 0.2, 0.3, 0.4] and r3[2] is None, str(r3))

chk("空文本 → 全 None", P.parse_multibox("no json here", 2) == [None, None])
chk("_scale4 与 parse_boxes 同源一致",
    P._scale4(10, 20, 30, 40)
    == P.parse_boxes('[{"bbox_2d": [10, 20, 30, 40]}]')[0])

# ---------------- ② 开关全开 ----------------
print("=== ② 开关全开（M 件配置）===")
os.environ.update({"QWEN_RULES5": "1", "QWEN_MULTIBOX": "1",
                   "QWEN_MODALITIES": "3", "QWEN_MAXNEW": "1024",
                   "QWEN_THINKING": "0", "QWEN_ENGINE": "api32b"})
P = importlib.reload(P)
chk("MAXNEW=1024", P.MAXNEW == 1024, str(P.MAXNEW))
chk("RULES_BLOCK = RULES5_TEXT + 换行",
    P.RULES_BLOCK == P.RULES5_TEXT + "\n")
q = "the second cup from the left"
pf = P.FULL_DIRECT_PROMPT.format(q=q, rules=P.RULES_BLOCK)
pt = P.THREE_IMG_PROMPT.format(q=q, rules=P.RULES_BLOCK)
anchor = "Description: %s\n" % q
chk("规则落在 Description 之后（FULL）",
    (anchor + P.RULES_BLOCK + "Output ONLY") in pf)
chk("规则落在 Description 之后（THREE）",
    (anchor + P.RULES_BLOCK + "Output ONLY") in pt)
chk("5 条规则编号齐全",
    all(("%d. " % i) in P.RULES5_TEXT for i in range(1, 6)))
chk("断言在开关全开后仍成立",
    P.THREE_IMG_PROMPT.split("{q}")[1]
    == P.FULL_DIRECT_PROMPT.split("{q}")[1])
mb = P.MULTIBOX_PROMPT.format(scene=P.SCENE3, where="the visible-light image "
                              "(image 1)", n=2, items="1. a\n2. b",
                              rules=P.RULES_BLOCK)
chk("MULTIBOX 含 n/items/scene/where 四处占位展开",
    "exactly 2 objects" in mb and "1. a" in mb and "image 1" in mb)
chk("解析在开关全开后仍正确",
    P.parse_multibox(ok, 3)[2] == [0.05, 0.06, 0.07, 0.08])

# ---------------- ④ 合并图模式（QWEN_MODALITIES="3m"） ----------------
print("=== ④ 合并图模式 3m ===")
os.environ.update({"QWEN_MODALITIES": "3m", "QWEN_MERGE_SPACE": "panel"})
P = importlib.reload(P)
w, h = 40, 24
vis = Image.new("RGB", (w, h), (255, 0, 0))
ir = Image.new("RGB", (w, h), (0, 255, 0))
dep = Image.new("RGB", (w, h), (0, 0, 255))
mg = P.merge3(vis, ir, dep, sep=4)
chk("merge3 尺寸 = 3W + 2·sep × H",
    mg.size == (w * 3 + 8, h), str(mg.size))
chk("merge3 左格为可见光", mg.getpixel((1, 1)) == (255, 0, 0))
chk("merge3 中格为红外", mg.getpixel((w + 5, 1)) == (0, 255, 0))
chk("merge3 右格为深度", mg.getpixel((2 * (w + 4) + 1, 1)) == (0, 0, 255))
chk("MERGE_SPACE 默认 panel（不换算）",
    P.MERGE_SPACE == "panel"
    and P.remap_merged([0.1, 0.2, 0.3, 0.4]) == [0.1, 0.2, 0.3, 0.4])
sc, wh_ = P.scene_where()
chk("scene_where 在 3m 下给合并图说法",
    sc == P.SCENE_MERGED and "LEFT (visible-light) panel" in wh_)

os.environ["QWEN_MERGE_SPACE"] = "whole"
P = importlib.reload(P)
chk("MERGE_SPACE=whole 时 x 乘 3",
    P.remap_merged([0.1, 0.2, 0.3, 0.4]) == [0.3, 0.2, 0.8999999999999999,
                                              0.4]
    or [round(v, 6) for v in P.remap_merged([0.1, 0.2, 0.3, 0.4])]
    == [0.3, 0.2, 0.9, 0.4])
chk("MERGE_SPACE=whole 时 x 越界被裁到 1.0",
    P.remap_merged([0.5, 0.1, 0.9, 0.2])[2] == 1.0)

print()
if fails:
    print("★失败 %d / %d 项：%s" % (len(fails), nchk, fails))
    sys.exit(1)
print("✓ 全部通过（%d 项）" % nchk)

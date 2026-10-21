# -*- coding: utf-8 -*-
"""构建序数层二次枚举清单（逐图定制 prompt），输出 ordinal_retry_manifest.json。"""
import json
import os

BOX = "<box><x1><y1><x2><y2></box>"
TEST = os.path.join(
    r"d:\Desktop\2026AiC\初赛数据集-基于大模型的多模态视觉理解与推理",
    "test.json")


def make_entry(tag, qid, core, prompt):
    """生成单条清单项。"""
    test_data = json.load(open(TEST, encoding="utf-8"))
    return tag, {
        "tag": tag,
        "visible": test_data[qid]["visible"],
        "core": core,
        "prompt": prompt,
        "queries": [qid],
    }


def main():
    """组装 9 条定制枚举措辞。"""
    items = [
        make_entry(
            "001773", "001773_005", "paddle",
            f"Locate every table tennis paddle in this image. Output one {BOX} "
            "for EACH paddle from left to right, including partially visible ones."),
        make_entry(
            "010019", "010019_003", "person",
            f"Locate every person in this image, including very small distant "
            f"people standing behind the railing or among trees. Output one {BOX} "
            "for EACH person from left to right. Do not skip any person."),
        make_entry(
            "016774", "016774_001", "person",
            f"Locate every person in this image, including very small distant "
            f"pedestrians on the sidewalk. Output one {BOX} for EACH person from "
            "left to right. Do not skip any person."),
        make_entry(
            "010033a", "010033_004", "bear",
            f"Locate every bear in this image, including the bear in the water "
            f"pool and bears standing in shadow. Output one {BOX} for EACH bear "
            "from left to right."),
        make_entry(
            "010033b", "010033_004", "bear",
            "How many bears (brown bears and black bears) are in this image? "
            f"Output one {BOX} for EACH bear from left to right including "
            "partially visible ones."),
        make_entry(
            "014716a", "014716_003", "diamond road marking",
            "Locate every diamond-shaped white road marking painted on the "
            f"asphalt ground. Output one {BOX} for EACH diamond marking from "
            "left to right."),
        make_entry(
            "014716b", "014716_003", "diamond road sign",
            "Locate every diamond shaped sign lying on the ground (white "
            f"outlined diamonds painted on the road surface). Output one {BOX} "
            "for EACH from left to right."),
        make_entry(
            "012826", "012826_001", "sofa cushion",
            "Locate every individual sofa seat cushion in this image; the sofa "
            f"has several separate cushions. Output one tight {BOX} for EACH "
            "separate cushion from left to right."),
        make_entry(
            "016333", "016333_002", "white pelican",
            f"Locate every WHITE pelican in this image. Do not include the "
            f"black swan swimming in the water. Output one {BOX} for EACH white "
            "pelican from left to right."),
    ]
    manifest = dict(items)
    out = r"d:\Desktop\2026AiC\ordinal_retry_manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"retry tags: {len(manifest)}")
    for tag, info in manifest.items():
        print(f"  {tag} -> {info['queries']}")


if __name__ == "__main__":
    main()
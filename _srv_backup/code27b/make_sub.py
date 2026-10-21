import json, os
queries = json.load(open(r'D:\GroundingDino\queries_ref.json', encoding='utf-8'))
preds = json.load(open(r'D:\GroundingDino\v27b\predictions_27b.json', encoding='utf-8'))

out = {}
for k, qv in queries.items():
    bbox = preds.get(k, [0.49,0.49,0.51,0.51])
    out[k] = {
        "visible": qv["visible"],
        "infrared": qv["infrared"],
        "depth": qv["depth"],
        "query": qv["query"],
        "bbox": [round(x,4) for x in bbox]
    }

with open(r'D:\GroundingDino\v27b\submission.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"submission: {len(out)} entries")

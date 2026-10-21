/root/miniconda3/envs/myconda/bin/python - <<'PYEOF'
import json
def L(p): return json.load(open(p))
L2 = L('/root/output/predictions_v2_highres_slow.json')
M  = L('/root/output/result_gd_3modal_v2.json')
G  = L('/root/output/predictions_grounding.json')

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2-x1), max(0.0, y2-y1)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 1e-9 else 0.0

CONF = {
 "000282_004": [0.2021,0.3015,0.3054,0.4004],
 "000282_008": [0.0312,0.3244,0.1368,0.5068],
 "002004_001": [0.4208,0.2913,0.4667,0.4365],
 "002004_002": [0.4543,0.5115,0.5251,0.6628],
}
print("4 个已确认样本在两种口径下的表现：")
for k, gt in CONF.items():
    l2, m, g = L2[k]['bbox'], M[k]['bbox'], G[k]['bbox']
    repl = (iou(l2, g) >= 0.5 and iou(l2, m) < 0.5 and iou(g, m) < 0.5)
    fused = [0.6*l2[i] + 0.4*g[i] for i in range(4)]
    print(f"  {k}: 基线IoU={iou(m,gt):.3f} 保守规则替换={repl} "
          f"替换后IoU={iou(fused,gt) if repl else iou(m,gt):.3f} "
          f"(IoU(L2,G)={iou(l2,g):.3f}, IoU(L2,M)={iou(l2,m):.3f}, IoU(G,M)={iou(g,m):.3f})")
PYEOF

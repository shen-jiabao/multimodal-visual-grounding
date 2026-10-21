/root/miniconda3/envs/myconda/bin/python - <<'PYEOF'
import json, re
OUT='/root/output'
L2=json.load(open(f'{OUT}/predictions_v2_highres_slow.json'))
M =json.load(open(f'{OUT}/result_gd_3modal_v2.json'))
G =json.load(open(f'{OUT}/predictions_grounding.json'))
F =json.load(open(f'{OUT}/fusion_g4.json'))
Q =json.load(open('/root/infer/queries.json'))
def area(b): return (b[2]-b[0])*(b[3]-b[1])
def iou(a,b):
    x1,y1,x2,y2=max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])
    iw,ih=max(0.,x2-x1),max(0.,y2-y1); it=iw*ih; ua=area(a)+area(b)-it
    return it/ua if ua>1e-9 else 0.
ORD = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)\b", re.I)
SPAT= re.compile(r"leftmost|rightmost|topmost|bottommost|nearest|closest|farthest|furthest|"
                 r"\bleft\b|\bright\b|\btop\b|\bbottom\b|behind|in front|above|below|"
                 r"beside|next to|between|foreground|background|upper|lower", re.I)
keys=sorted(set(L2)&set(M)&set(G))

def stats(name, cond):
    sel=[k for k in keys if cond(k)]
    chg=[k for k in sel if iou(L2[k]['bbox'], F[k]['bbox'])<0.999]
    big=sum(1 for k in sel if area(M[k]['bbox'])>0.35)
    small_base=sum(1 for k in sel if area(M[k]['bbox'])<0.1)
    tiny_l2=sum(1 for k in chg if area(L2[k]['bbox'])<0.01)
    print(f"{name:34s} 命中={len(sel):4d} 会改={len(chg):4d} "
          f"基线>0.35={big:4d} 基线<0.1={small_base:3d} L2<0.01={tiny_l2:4d}")

print("=== R1 细分（均为 基线面积>0.2 前提下）===")
stats("仅序数词", lambda k: bool(ORD.search(Q[k]['query'])) and area(M[k]['bbox'])>0.2)
stats("仅纯空间词(非序数)", lambda k: (not ORD.search(Q[k]['query'])) and bool(SPAT.search(Q[k]['query'])) and area(M[k]['bbox'])>0.2)
stats("序数或空间(即 R1 全集)", lambda k: (ORD.search(Q[k]['query']) or SPAT.search(Q[k]['query'])) and area(M[k]['bbox'])>0.2)
print()
stats("R1 基础上 基线>0.35", lambda k: (ORD.search(Q[k]['query']) or SPAT.search(Q[k]['query'])) and area(M[k]['bbox'])>0.35)
stats("R1 基础上 基线>0.25", lambda k: (ORD.search(Q[k]['query']) or SPAT.search(Q[k]['query'])) and area(M[k]['bbox'])>0.25)

# 各档基线面积下、M 与 L2 的重合度（判断基线是否"离谱"）
print("\n=== R1 命中样本里 IoU(M,L2) 分布 ===")
import collections
c=collections.Counter()
for k in keys:
    if (ORD.search(Q[k]['query']) or SPAT.search(Q[k]['query'])) and area(M[k]['bbox'])>0.2:
        v=iou(M[k]['bbox'], L2[k]['bbox'])
        c['=0' if v<1e-9 else ('<0.1' if v<0.1 else ('0.1~0.5' if v<0.5 else '>=0.5'))]+=1
print(dict(c))
PYEOF

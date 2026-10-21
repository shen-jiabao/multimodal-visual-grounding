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
    iw,ih=max(0.,x2-x1),max(0.,y2-y1); it=iw*ih
    ua=area(a)+area(b)-it
    return it/ua if ua>1e-9 else 0.

ORD = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)\b", re.I)
SPAT= re.compile(r"leftmost|rightmost|topmost|bottommost|nearest|closest|farthest|furthest|"
                 r"\bleft\b|\bright\b|\btop\b|\bbottom\b|behind|in front|above|below|"
                 r"beside|next to|between|foreground|background|upper|lower", re.I)
COLL= re.compile(r"all the|group of|crowd|people|men|women|children|students|tourists|"
                 r"\bpair\b|\btwo\b|\bthree\b|\bfour\b|\bfive\b|\bboth\b", re.I)

keys=sorted(set(L2)&set(M)&set(G))
CONF=["000282_004","000282_008","002004_001","002004_002"]
JUDGED=["000023_002","000023_003","000023_004","000023_006","000478_005"]

# ---- 候选规则 R1: 序数/空间类 且 基线面积>0.2 → 改用 L2 ----
R1=[k for k in keys if (ORD.search(Q[k]['query']) or SPAT.search(Q[k]['query']))
    and area(M[k]['bbox'])>0.2]
R1_chg=[k for k in R1 if iou(L2[k]['bbox'], M[k]['bbox'])<0.999]
big_now=sum(1 for k in keys if area(F[k]['bbox'])>0.35)
sim={}
for k in keys:
    sim[k]= L2[k]['bbox'] if k in R1_chg else F[k]['bbox']
big_r1=sum(1 for k in keys if area(sim[k])>0.35)
print("=== R1: 序数/空间 且 基线面积>0.2 → 用 L2 ===")
print(f"  命中样本 {len(R1)}，其中相对 G4 会改变 {len(R1_chg)}")
print(f"  大框: G4 {big_now} → R1 {big_r1}")
print(f"  护栏样本是否被波及: {[(k, k in R1_chg) for k in CONF]}")
print(f"  判定样本结果: " + ", ".join(
    f"{k}: M={area(M[k]['bbox']):.3f}→L2={area(L2[k]['bbox']):.3f}" for k in JUDGED[:4]))

# ---- 候选规则 R2: 集合/复数类 → 禁止"缩小"替换 ----
rep=[k for k in keys if iou(F[k]['bbox'], M[k]['bbox'])<0.999]
shrink=[k for k in rep if area(F[k]['bbox'])<area(M[k]['bbox'])]
R2=[k for k in shrink if COLL.search(Q[k]['query'])]
print("\n=== R2: 集合/复数类 禁止缩小替换(回退较大框) ===")
print(f"  G4 替换中缩小者 {len(shrink)}，其中含集合/复数词 {len(R2)}")
sim2={k:(M[k]['bbox'] if k in R2 else F[k]['bbox']) for k in keys}
big_r2=sum(1 for k in keys if area(sim2[k])>0.35)
print(f"  大框: G4 {big_now} → R2 {big_r2}")
print(f"  护栏样本是否被波及: {[(k, k in R2) for k in CONF]}")
print(f"  判定样本 000478_005: M={area(M['000478_005']['bbox']):.3f} "
      f"G4={area(F['000478_005']['bbox']):.3f} → R2回退后={area(M['000478_005']['bbox']):.3f}")

# ---- R1+R2 合并影响 ----
both={}
for k in keys:
    b=F[k]['bbox']
    if k in R1_chg: b=L2[k]['bbox']
    if k in R2: b=M[k]['bbox']
    both[k]=b
chg_both=[k for k in keys if iou(both[k], F[k]['bbox'])<0.999]
big_both=sum(1 for k in keys if area(both[k])>0.35)
print("\n=== R1+R2 合并 ===")
print(f"  相对 G4 改变 {len(chg_both)} 条；大框 {big_now} → {big_both}")
print(f"  护栏样本: {[(k, k in chg_both) for k in CONF]}")
PYEOF

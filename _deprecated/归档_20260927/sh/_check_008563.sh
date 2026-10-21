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
    iw,ih=max(0.,x2-x1),max(0.,y2-y1); inter=iw*ih
    ua=area(a)+area(b)-inter
    return inter/ua if ua>1e-9 else 0.

k='008563_003'
print('query :', Q[k]['query'])
for name,d in [('L2',L2),('M_base',M),('G',G),('G4_final',F)]:
    b=d[k]['bbox']
    print(f'{name:9s} {[round(v,4) for v in b]}  area={area(b):.4f}')
print('IoU(L2,M) =',round(iou(L2[k]["bbox"],M[k]["bbox"]),4))
print('IoU(L2,G) =',round(iou(L2[k]["bbox"],G[k]["bbox"]),4))
print('IoU(M ,G) =',round(iou(M[k]["bbox"],G[k]["bbox"]),4))
print('G4是否替换 =', F[k]['bbox'] != M[k]['bbox'])

# ===== 按 query 类型禁替换的影响量化 =====
ORDS = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)\b"
                  r"|\b(two|three|four|five|six|seven|eight|nine|ten|pair|both)\b"
                  r"|from (the )?(left|right) to (the )?(left|right)", re.I)
SPAT = re.compile(r"leftmost|rightmost|topmost|bottommost|nearest|closest|farthest|furthest"
                  r"|\bleft\b|\bright\b|\btop\b|\bbottom\b|behind|in front|above|below"
                  r"|beside|next to|between|foreground|background|upper|lower", re.I)

keys=sorted(set(L2)&set(M)&set(G))
# G4 已替换集合
rep=[kk for kk in keys if F[kk]['bbox']!=M[kk]['bbox']]
blocked=[kk for kk in rep if ORDS.search(Q[kk]['query']) or SPAT.search(Q[kk]['query'])]
print(f"\n=== 按 query 类型禁替换的影响 ===")
print(f"G4 当前替换: {len(rep)}")
print(f"  其中含序数/空间词被拦下: {len(blocked)} ({100*len(blocked)/len(rep):.1f}%)")
print(f"  禁替换后剩余替换: {len(rep)-len(blocked)}")
big0=sum(1 for kk in keys if area(M[kk]['bbox'])>0.35)
big_new=sum(1 for kk in keys if (area(M[kk]['bbox']) if kk in blocked else area(F[kk]['bbox']))>0.35)
fixed=sum(1 for kk in rep if kk not in blocked and area(M[kk]['bbox'])>0.35 and area(F[kk]['bbox'])<=0.35)
print(f"  大框: 基线 {big0} → 禁替换后 {big_new} (修掉 {fixed})")
# 也看 885 候选集里被拦下的比例
c885=json.load(open(f'{OUT}/fusion_conservative.json'))
c885_ch=[kk for kk in keys if c885[kk]['bbox']!=M[kk]['bbox']]
b2=[kk for kk in c885_ch if ORDS.search(Q[kk]['query']) or SPAT.search(Q[kk]['query'])]
print(f"参考: 885 候选里含序数/空间词的 {len(b2)}/{len(c885_ch)} ({100*len(b2)/len(c885_ch):.1f}%)")
PYEOF

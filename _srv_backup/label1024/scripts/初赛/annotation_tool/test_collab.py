import http.cookiejar
import json
import os
import urllib.error
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8765"
OUT = os.path.join("scripts", "初赛", "annotation_tool", "outputs", "annotations_test3")

class Client:
    def __init__(self):
        self.jar = http.cookiejar.CookieJar()
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))

    def _open(self, path, data=None, parse=True):
        headers = {"Content-Type": "application/json"} if data is not None else {}
        req = urllib.request.Request(BASE + path, data=data, headers=headers)
        try:
            with self.op.open(req) as r:
                return r.status, (json.load(r) if parse else r.read())
        except urllib.error.HTTPError as e:
            return e.code, (json.load(e) if parse else e.read())

    def get(self, path):
        return self._open(path)

    def get_bytes(self, path):
        return self._open(path, parse=False)

    def post(self, path, body=None):
        return self._open(path, data=json.dumps(body or {}).encode())

results = []
def check(name, cond, extra=""):
    results.append((name, cond, extra))
    print(("PASS " if cond else "FAIL ") + name + (("  " + extra) if extra else ""))

A, B = Client(), Client()
st, s1 = A.get("/api/session"); check("S1 session", st == 200 and not s1["is_admin"] and s1["name"].startswith("标注员-"), s1["name"])
st, s2 = B.get("/api/session"); check("S2 session", st == 200 and s2["name"] != s1["name"], s2["name"])

_, r1 = A.post("/api/claim", {"image": "Images/visible/000002.png"})
check("S1 claim", r1["claimed"] and r1["image"] == "Images/visible/000002.png", r1.get("image", ""))
imgA = r1["image"]
_, r2 = B.post("/api/claim", {"image": "Images/visible/000023.png"})
check("S2 claim 另一张图", r2["claimed"] and r2["image"] == "Images/visible/000023.png" and r2["image"] != imgA, r2.get("image", ""))
imgB = r2["image"]

C = Client()
_, s3 = C.get("/api/session")
_, r3 = C.post("/api/claim", {"image": "Images/visible/000044.png"})
check("S3 指定认领未认领图", r3["claimed"] and r3["image"] == "Images/visible/000044.png", r3.get("error", ""))
imgC = r3["image"]
_, r = A.post("/api/claim", {"image": imgC})
check("S1 已有在手图时指定认领 → already", r.get("already") and r["image"] == imgA, str(r)[:80])
D = Client()
_, s4 = D.get("/api/session")
_, r = D.post("/api/claim", {"image": imgA})
check("S4 无在手图指定他人图 → 拒绝", not r["claimed"] and "标注中" in r.get("error", ""), r.get("error", ""))
_, r = D.post("/api/claim", {"image": "not_exist.png"})
check("S4 指定不存在图 → 拒绝", not r["claimed"] and "不存在" in r.get("error", ""), r.get("error", ""))

_, g = A.get("/api/group?image=" + urllib.parse.quote(imgA))
check("S1 group can_edit", g["can_edit"] and g["owner"] == s1["name"])
_, g2 = B.get("/api/group?image=" + urllib.parse.quote(imgA))
check("S2 看 S1 的图只读", not g2["can_edit"] and g2["owner"] == s1["name"])

qidA = g["queries"][0]["qid"]
st, r = B.post("/api/save", {"qid": qidA, "bbox": [0.1, 0.1, 0.3, 0.3]})
check("S2 保存 S1 的图 → 拒绝", st == 400 and r.get("forbidden"), str(r))

_, r = A.post("/api/save", {"qid": qidA, "bbox": [0.1, 0.1, 0.3, 0.3]})
check("S1 保存自己的图", "error" not in r and r["stats"]["done"] == 1, json.dumps(r, ensure_ascii=False)[:100])
tmpA = os.path.join(OUT, "tmp_" + s1["name"] + ".json")
check("S1 临时文件存在", os.path.exists(tmpA))
rec = json.load(open(tmpA, encoding="utf-8"))
check("临时文件内容正确", rec["image"] == imgA and qidA in rec["data"])

_, gB = B.get("/api/group?image=" + urllib.parse.quote(imgB))
qidB = gB["queries"][0]["qid"]
_, r = B.post("/api/save", {"qid": qidB, "bbox": [0.2, 0.2, 0.4, 0.4]})
check("S2 保存自己的图", "error" not in r, json.dumps(r, ensure_ascii=False)[:100])

_, r = B.post("/api/heartbeat")
check("S2 heartbeat 返回自己图", r["image"] == imgB)

_, r = A.post("/api/release")
check("S1 交还", r["released"] and r["image"] == imgA)
check("S1 临时文件已删", not os.path.exists(tmpA))
_, r = A.post("/api/claim", {"image": "Images/visible/000037.png"})
check("S1 重新认领", r["claimed"] and r["image"] == "Images/visible/000037.png", r.get("image", ""))

store = json.load(open(os.path.join(OUT, "annotations.json"), encoding="utf-8"))
check("交还合并进主文件", store.get(qidA, {}).get("status") == "done")

_, r = A.post("/api/admin_release", {"worker": s2["name"]})
check("管理员强制释放", r["released"] and r["image"] == imgB, json.dumps(r, ensure_ascii=False)[:100])
store = json.load(open(os.path.join(OUT, "annotations.json"), encoding="utf-8"))
check("强制释放保留标注（合并）", store.get(qidB, {}).get("status") == "done")
_, roster = A.get("/api/roster")
s2row = [x for x in roster["rows"] if x["name"] == s2["name"]][0]
check("强制释放后空闲", s2row["image"] is None)
check("释放日志记录", any(l["reason"] == "admin" for l in roster["log"]))
check("roster 含未认领数", roster["unclaimed_images"] >= 1)

_, exp = A.get("/api/export")
check("导出数量不完整 → 拒绝", "error" in exp and "不完整" in exp.get("error", ""), str(exp)[:100])
check("导出拒绝时未生成文件", not os.path.exists(os.path.join(OUT, "answer_full.zip")))

st, ro = A.get_bytes("/roster.html")
check("roster 页面可访问", st == 200 and "认领名单".encode() in ro)
st, u = A.get_bytes("/")
check("主页面可访问", st == 200 and "真值标注工具".encode() in u)

_, s = A.get("/api/session")
check("cookie 会话持久（同名）", s["name"] == s1["name"])

_, r = A.post("/api/admin_delete_session", {"worker": "不存在的会话"})
check("管理员删除不存在会话 → 拒绝", not r["deleted"] and "不存在" in r.get("error", ""), r.get("error", ""))

_, r = A.post("/api/admin_delete_session", {"worker": s4["name"]})
check("管理员删除无在手图会话", r["deleted"] and r["image"] is None, json.dumps(r, ensure_ascii=False)[:100])
_, roster = A.get("/api/roster")
check("删除后名单移除", all(x["name"] != s4["name"] for x in roster["rows"]))

_, gC = C.get("/api/group?image=" + urllib.parse.quote(imgC))
qidC = gC["queries"][0]["qid"]
_, r = C.post("/api/save", {"qid": qidC, "bbox": [0.3, 0.3, 0.5, 0.5]})
check("S3 保存自己的图（供删除验证合并）", "error" not in r, json.dumps(r, ensure_ascii=False)[:100])
tmpC = os.path.join(OUT, "tmp_" + s3["name"] + ".json")
check("S3 临时文件存在", os.path.exists(tmpC))
_, r = A.post("/api/admin_delete_session", {"worker": s3["name"]})
check("管理员删除有在手图会话（先释放合并）", r["deleted"] and r["image"] == imgC, json.dumps(r, ensure_ascii=False)[:100])
store = json.load(open(os.path.join(OUT, "annotations.json"), encoding="utf-8"))
check("删除前标注已合并进主文件", store.get(qidC, {}).get("status") == "done")
check("删除后临时文件已删", not os.path.exists(tmpC))
_, roster = A.get("/api/roster")
check("删除后名单移除（含在手图会话）", all(x["name"] != s3["name"] for x in roster["rows"]))
check("删除后原在手图空闲", all(x["image"] != imgC for x in roster["rows"]))

_, r = A.post("/api/admin_delete_session", {"worker": s1["name"]})
check("删除自己 → 拒绝", not r["deleted"] and "管理员" in r.get("error", ""), r.get("error", ""))

E = Client()
_, s5 = E.get("/api/session")
_, roster0 = A.get("/api/roster")
occupied0 = {x["image"] for x in roster0["rows"] if x["status"] == "active"}
_, r = E.post("/api/claim")
check("随机认领未标注且未认领", r["claimed"] and r["image"] not in occupied0, r.get("image", ""))
_, gE = E.get("/api/group?image=" + urllib.parse.quote(r["image"]))
check("认领图为未标注图", any(x["status"] == "pending" for x in gE["queries"]))

_, roster = A.get("/api/roster")
occupied = {x["image"] for x in roster["rows"] if x["status"] == "active"}
all_imgs = ["Images/visible/%s.png" % n for n in ("000002", "000023", "000037", "000044", "000051", "000058")]
free = [img for img in all_imgs if img not in occupied]
F = Client()
_, s6 = F.get("/api/session")
_, r = F.post("/api/claim_back")
check("F 无历史回退 → 拒绝", not r["backed"] and "历史" in r.get("error", ""), r.get("error", ""))
_, r = F.post("/api/claim", {"image": free[0]})
imgF1 = r["image"]
check("F 认领第一张", r["claimed"] and r["image"] == free[0], r.get("error", ""))
_, r = F.post("/api/release")
check("F 交还第一张", r["released"] and r["image"] == imgF1, r.get("error", ""))
_, r = F.post("/api/claim", {"image": free[1]})
imgF2 = r["image"]
check("F 认领第二张", r["claimed"] and r["image"] == free[1], r.get("error", ""))
_, r = F.post("/api/release")
check("F 交还第二张", r["released"] and r["image"] == imgF2, r.get("error", ""))
_, r = F.post("/api/claim_back")
check("F 无在手回退到最近认领", r["backed"] and r["image"] == imgF2, r.get("error", ""))
_, r = F.post("/api/claim_back")
check("F 回退到上一认领", r["backed"] and r["image"] == imgF1, r.get("error", ""))
_, g = F.get("/api/group?image=" + urllib.parse.quote(imgF1))
check("F 回退后可编辑", g["can_edit"] and g["owner"] == s6["name"])
_, b = F.get("/api/bootstrap")
check("F 回退后栈仅剩当前不可再回退", not b["collab"]["can_back"])
_, r = F.post("/api/claim_back")
check("F 连续回退无更早记录 → 拒绝", not r["backed"] and "更早" in r.get("error", ""), r.get("error", ""))

fails = [x for x in results if not x[1]]
print("\n== %d/%d PASS ==" % (len(results) - len(fails), len(results)))
if fails:
    for f in fails: print("FAILED:", f[0], f[2])

import json
import os
import re
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCHIVE = os.path.join(ROOT, "outputs", "复赛", "archive")
IMG_DIR = os.path.join(ROOT, "dataset", "Images")
QUERIES_PATH = os.path.join(ROOT, "dataset", "queries", "queries.json")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
AUDIT_DIR = os.path.join(ROOT, "outputs", "复赛", "audit")

RUN_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
SAMPLE_RE = re.compile(r"^\d{6}_\d{3}$")
MODALS = ("visible", "infrared", "depth")

_queries = None
_dict = None


def load_queries():
    global _queries
    if _queries is None:
        with open(QUERIES_PATH, encoding="utf-8") as f:
            _queries = json.load(f)
    return _queries


def load_dict():
    global _dict
    if _dict is None:
        with open(os.path.join(STATIC_DIR, "query_zh.json"), encoding="utf-8") as f:
            _dict = json.load(f)
    return _dict


def list_runs():
    runs = []
    if os.path.isdir(ARCHIVE):
        for name in sorted(os.listdir(ARCHIVE)):
            if not RUN_RE.fullmatch(name):
                continue
            rdir = os.path.join(ARCHIVE, name)
            if not os.path.isdir(rdir):
                continue
            runs.append(
                {
                    "name": name,
                    "predictions": os.path.isfile(os.path.join(rdir, "predictions.json")),
                    "submission": os.path.isfile(os.path.join(rdir, "submission.json")),
                }
            )
    return runs


def run_dir(run):
    if not RUN_RE.fullmatch(run or ""):
        return None
    d = os.path.join(ARCHIVE, run)
    return d if os.path.isdir(d) else None


def load_boxes(run, src):
    d = run_dir(run)
    if d is None:
        return None
    path = os.path.join(d, "predictions.json" if src == "predictions" else "submission.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if src == "submission":
        data = {k: v.get("bbox") for k, v in data.items()}
    return data


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _text(self, text, code=200):
        self._send(code, text.encode("utf-8"), "text/html; charset=utf-8")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        if path == "/":
            self._serve_static("index.html")
            return
        if path.startswith("/static/"):
            self._serve_static(path[len("/static/"):])
            return
        if path == "/api/runs":
            self._json({"runs": list_runs()})
            return
        if path == "/api/meta":
            self._api_meta(qs)
            return
        if path == "/api/sample":
            self._api_sample(qs)
            return
        if path.startswith("/img/"):
            self._serve_image(path)
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/annotations":
            self._api_annotations()
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def _serve_static(self, name):
        name = os.path.basename(name)
        path = os.path.join(STATIC_DIR, name)
        if not os.path.isfile(path):
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        if name.endswith(".html"):
            ctype = "text/html; charset=utf-8"
        elif name.endswith(".json"):
            ctype = "application/json; charset=utf-8"
        else:
            ctype = "application/octet-stream"
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    def _serve_image(self, path):
        parts = path.split("/")
        if len(parts) != 4 or parts[1] != "img":
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        modal = parts[2]
        fname = parts[3]
        m = re.fullmatch(r"(\d{6})\.png", fname)
        if modal not in MODALS or not m:
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        img = os.path.join(IMG_DIR, modal, fname)
        if not os.path.isfile(img):
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        with open(img, "rb") as f:
            self._send(200, f.read(), "image/png")

    def _api_meta(self, qs):
        run = (qs.get("run") or [""])[0]
        d = run_dir(run)
        if d is None:
            self._json({"error": "unknown run"}, 404)
            return
        boxes = load_boxes(run, "predictions")
        ids = list(boxes.keys()) if boxes else []
        q = (qs.get("q") or [""])[0].strip()
        if q:
            queries = load_queries()
            ql = q.lower()
            ids = [i for i in ids if ql in queries.get(i, {}).get("query", "").lower()]
        self._json({"run": run, "total": len(ids), "ids": ids})

    def _api_sample(self, qs):
        run = (qs.get("run") or [""])[0]
        sid = (qs.get("id") or [""])[0]
        src = (qs.get("src") or ["predictions"])[0]
        if src not in ("predictions", "submission"):
            src = "predictions"
        if run_dir(run) is None:
            self._json({"error": "unknown run"}, 404)
            return
        if not SAMPLE_RE.fullmatch(sid):
            self._json({"error": "bad id"}, 400)
            return
        queries = load_queries()
        q = queries.get(sid)
        if q is None:
            self._json({"error": "sample not in dataset"}, 404)
            return
        zh = load_dict().get(q["query"])
        result = {
            "id": sid,
            "query": q["query"],
            "query_zh": zh,
            "images": {m: "/img/%s/%s.png" % (m, sid.split("_")[0]) for m in MODALS},
            "boxes": {},
        }
        primary = load_boxes(run, src)
        if primary is None:
            self._json({"error": "no boxes for run"}, 404)
            return
        result["boxes"][run] = primary.get(sid)
        for other in (qs.get("compare") or [""])[0].split(","):
            other = other.strip()
            if not other:
                continue
            if run_dir(other) is None:
                result["boxes"][other] = None
                continue
            boxes = load_boxes(other, src)
            result["boxes"][other] = boxes.get(sid) if boxes else None
        self._json(result)

    def _api_annotations(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 10 * 1024 * 1024:
            self._json({"error": "bad body"}, 400)
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            self._json({"error": "invalid json"}, 400)
            return
        run = body.get("run")
        src = body.get("src", "predictions")
        items = body.get("items")
        if run_dir(run) is None or not isinstance(items, dict):
            self._json({"error": "bad payload"}, 400)
            return
        os.makedirs(AUDIT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = "manual_review_%s_%s.json" % (run, ts)
        out = os.path.join(AUDIT_DIR, fname)
        payload = {"run": run, "src": src, "generated_at": datetime.now().isoformat(timespec="seconds"), "items": items}
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        self._json({"saved": True, "path": out, "count": len(items)})


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("result_viewer http://127.0.0.1:%d" % port)
    for r in list_runs():
        print("  run:", r["name"], "pred=%s sub=%s" % (r["predictions"], r["submission"]))
    server.serve_forever()


if __name__ == "__main__":
    main()

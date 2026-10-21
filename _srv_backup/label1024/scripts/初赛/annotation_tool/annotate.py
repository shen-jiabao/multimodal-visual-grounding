from __future__ import annotations

import argparse
import base64
import io
import json
import msvcrt
import os
import re
import socket
import sys
import threading
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np
from PIL import Image

from collab import COOKIE_MAX_AGE, Collab

def find_project_root(start: str) -> str:
    d = os.path.abspath(start)
    while not os.path.isdir(os.path.join(d, "dataset")):
        parent = os.path.dirname(d)
        if parent == d:
            return d
        d = parent
    return d


PROJECT_ROOT = find_project_root(os.path.dirname(__file__))
UI_PATH = os.path.join(os.path.dirname(__file__), "annotate_ui.html")
MOBILE_UI_PATH = os.path.join(os.path.dirname(__file__), "annotate_ui_mobile.html")
ROSTER_PATH = os.path.join(os.path.dirname(__file__), "roster.html")
MOBILE_UA_MARKERS = ("mobile", "android", "iphone", "ipod", "windows phone")


def pick_ui_path(query: dict[str, list[str]], ua: str) -> str:
    view = (query.get("view") or [""])[0].lower()
    if view == "mobile":
        return MOBILE_UI_PATH
    if view == "desktop":
        return UI_PATH
    return MOBILE_UI_PATH if any(m in (ua or "").lower() for m in MOBILE_UA_MARKERS) else UI_PATH
TRANSLATIONS_PATH = os.path.join(os.path.dirname(__file__), "translations.json")
CANCELLED_PATH = os.path.join(os.path.dirname(__file__), "cancelled_queries.json")
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "annotations")
MAX_BODY_BYTES = 20 * 1024 * 1024


def parse_cookie(header: str | None, key: str) -> str | None:
    if not header:
        return None
    for part in header.split(";"):
        name, _, value = part.strip().partition("=")
        if name == key:
            return unquote(value)
    return None


def load_translations(path: str = TRANSLATIONS_PATH) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if isinstance(k, str) and v}
    except (OSError, ValueError):
        pass
    return {}


def load_cancelled(path: str = CANCELLED_PATH) -> set:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        qids = data.get("qids") if isinstance(data, dict) else None
        if isinstance(qids, list):
            return {str(q) for q in qids if q}
    except (OSError, ValueError):
        pass
    return set()


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def resolve_path(base: str) -> str:
    if not os.path.isabs(base):
        base = os.path.join(PROJECT_ROOT, base)
    return os.path.abspath(base)


def natural_key(path: str) -> int:
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    digits = re.sub(r"\D", "", stem)
    return int(digits) if digits else 0


def build_groups(queries: dict) -> list:
    by_image: dict[str, list] = {}
    for qid, item in queries.items():
        by_image.setdefault(item["visible"], []).append((qid, item))
    groups = [(image, sorted(items)) for image, items in by_image.items()]
    groups.sort(key=lambda g: natural_key(g[0]))
    return groups


def normalize_bbox(raw) -> list | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        vals = [max(0.0, min(1.0, float(v))) for v in raw]
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = vals
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def bbox_from_entry(entry) -> list | None:
    if isinstance(entry, dict):
        bbox = entry.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return list(bbox)
        return None
    if isinstance(entry, (list, tuple)) and len(entry) == 4:
        return list(entry)
    return None


def iou(box_a: list, box_b: list) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def score_from_dict(preds: dict, answers_path: str, out_dir: str,
                    threshold: float, source: str) -> dict:
    if not os.path.exists(answers_path):
        return {"error": f"答案文件不存在: {answers_path}，请先完成标注并导出答案"}
    with open(answers_path, "r", encoding="utf-8") as f:
        answers = json.load(f)
    details = []
    for qid in sorted(set(preds) & set(answers)):
        pb = bbox_from_entry(preds[qid])
        gb = bbox_from_entry(answers[qid])
        if pb is None or gb is None:
            continue
        v = iou(pb, gb)
        details.append({"qid": qid, "pred": pb, "gt": gb,
                        "iou": round(v, 4), "hit": v >= threshold})
    total = len(details)
    correct = sum(1 for d in details if d["hit"])
    mean_iou = float(np.mean([d["iou"] for d in details])) if total else 0.0
    errors = sorted((d for d in details if not d["hit"]), key=lambda d: d["iou"])
    report = {"predictions": source,
              "answers": os.path.abspath(answers_path),
              "threshold": threshold, "total": total, "correct": correct,
              "acc": round(correct / total, 4) if total else 0.0,
              "mean_iou": round(mean_iou, 4), "errors": errors}
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "score_report.json")
    write_json(report, report_path)
    return {"report": report, "report_file": report_path,
            "errors_head": errors[:20]}


def score_files(predictions_path: str, answers_path: str, out_dir: str,
                threshold: float = 0.5) -> dict:
    if not os.path.exists(predictions_path):
        return {"error": f"提交结果文件不存在: {predictions_path}"}
    with open(predictions_path, "r", encoding="utf-8") as f:
        preds = json.load(f)
    return score_from_dict(preds, answers_path, out_dir, threshold,
                           os.path.abspath(predictions_path))


def find_candidate_predictions(out_dir: str) -> list[str]:
    candidates = []
    roots = [out_dir, os.path.join(PROJECT_ROOT, "outputs")]
    for root in roots:
        for name in ("predictions.json", "submission.json"):
            path = os.path.join(root, name)
            if os.path.isfile(path) and path not in candidates:
                candidates.append(path)
    return candidates


def write_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_imported(data: dict) -> dict:
    normalized = {}
    for qid, raw in data.items():
        bbox = bbox_from_entry(raw)
        bbox = normalize_bbox(bbox)
        if bbox is not None:
            normalized[qid] = bbox
    return normalized


def detect_format(data: dict) -> str:
    for raw in data.values():
        if isinstance(raw, dict) and "visible" in raw and "bbox" in raw:
            return "full"
        break
    return "plain"


def load_import_content(b64: str, is_zip: bool) -> dict | None:
    try:
        raw = base64.b64decode(b64)
    except (ValueError, TypeError):
        return None
    try:
        if is_zip:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                names = zf.namelist()
                pick = next((n for n in ("submission.json", "answer.json", "predictions.json") if n in names), None)
                if pick is None:
                    jsons = [n for n in names if n.endswith(".json")]
                    pick = jsons[0] if jsons else None
                if pick is None:
                    return None
                data = json.loads(zf.read(pick).decode("utf-8"))
        else:
            data = json.loads(raw.decode("utf-8"))
    except (zipfile.BadZipFile, ValueError, OSError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def depth_jpeg(path: str) -> bytes:
    with Image.open(path) as img:
        arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    if hi - lo < 1e-3:
        hi = lo + 1e-3
    arr = np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=90)
    return buf.getvalue()


class Store:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        self.data: dict = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

    def get(self, qid: str) -> dict:
        return self.data.get(qid) or {}

    def _flush(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def save(self, qid: str, entry: dict) -> None:
        with self.lock:
            self.data[qid] = entry
            self._flush()

    def update_many(self, entries: dict) -> None:
        with self.lock:
            self.data.update(entries)
            self._flush()

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.data)


class App:
    def __init__(self, data_root: str, queries_file: str, out_dir: str,
                 max_images: int | None = None):
        self.data_root = resolve_path(data_root)
        self.root_abs = os.path.abspath(self.data_root)
        self.out_dir = resolve_path(out_dir)
        self.translations = load_translations()
        self.cancelled = load_cancelled()
        self._load_queries(os.path.join(self.data_root, queries_file), max_images)
        os.makedirs(self.out_dir, exist_ok=True)
        self.store = Store(os.path.join(self.out_dir, "annotations.json"))
        self._load_imported()
        self.collab = Collab(self.out_dir, self.store, self.groups, self.all_qids,
                             self.cancelled)

    def _load_queries(self, path: str, max_images: int | None) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.queries = json.load(f)
        self.groups = build_groups(self.queries)
        if max_images is not None:
            self.groups = self.groups[:max_images]
        self.all_qids = [qid for _, items in self.groups for qid, _ in items]
        self.image_index = {image: i for i, (image, _) in enumerate(self.groups)}
        self.qid_image = {qid: image for image, items in self.groups for qid, _ in items}

    def _load_imported(self) -> None:
        self.imported_path = os.path.join(self.out_dir, "imported_annotations.json")
        self.imported_name = ""
        self.imported = {}
        if os.path.exists(self.imported_path):
            with open(self.imported_path, "r", encoding="utf-8") as f:
                record = json.load(f)
            self.imported_name = record.get("name", "")
            self.imported = record.get("data", {}) if isinstance(record.get("data"), dict) else {}

    def import_predictions(self, body: dict) -> dict:
        data = body.get("data")
        if not isinstance(data, dict) or not data:
            b64 = body.get("file_b64")
            if not b64:
                return {"error": "导入文件为空或格式不正确"}
            data = load_import_content(b64, bool(body.get("is_zip")))
            if not data:
                return {"error": "文件解析失败：需为 JSON（qid→bbox）或 zip（内含 submission.json / answer.json）"}
        normalized = normalize_imported(data)
        if not normalized:
            return {"error": "未能从导入文件中解析出有效 bbox"}
        name = str(body.get("name") or "imported")
        write_json({"name": name, "data": normalized}, self.imported_path)
        self.imported_name = name
        self.imported = normalized
        known = set(self.all_qids)
        in_range = sum(1 for q in normalized if q in known)
        return {"name": name, "count": len(normalized),
                "format": detect_format(data),
                "covered": in_range,
                "out_of_range": len(normalized) - in_range,
                "invalid": len(data) - len(normalized)}

    def clear_imported(self) -> dict:
        if os.path.exists(self.imported_path):
            os.remove(self.imported_path)
        self.imported_name = ""
        self.imported = {}
        return {"cleared": True}

    def imported_payload(self) -> dict:
        return {"name": self.imported_name, "count": len(self.imported)}

    def stats(self) -> dict:
        return self.collab.merged_stats()

    def img_url(self, rel: str) -> str:
        return "/img?path=" + quote(rel)

    def group_payload(self, image: str, current_qid: str | None = None,
                      worker: str | None = None) -> dict | None:
        index = self.image_index.get(image)
        if index is None:
            return None
        items = self.groups[index][1]
        queries = []
        for qid, item in items:
            entry = self.store.get(qid)
            queries.append({"qid": qid, "query": item["query"],
                            "query_cn": self.translations.get(item["query"], ""),
                            "bbox": entry.get("bbox"),
                            "machine_bbox": self.imported.get(qid),
                            "status": entry.get("status", "pending"),
                            "cancelled": qid in self.cancelled})
        if current_qid is None:
            for qid, _ in items:
                if qid in self.cancelled:
                    continue
                if self.store.get(qid).get("status") != "done":
                    current_qid = qid
                    break
            if current_qid is None:
                for qid, _ in items:
                    if qid not in self.cancelled:
                        current_qid = qid
                        break
            if current_qid is None:
                current_qid = items[0][0]
        first = items[0][1]
        owner = self.collab.image_owner(image) if worker is not None else None
        return {"index": index, "total_groups": len(self.groups), "image": image,
                "prev_image": self.groups[index - 1][0] if index > 0 else None,
                "next_image": self.groups[index + 1][0] if index + 1 < len(self.groups) else None,
                "visible_url": self.img_url(first["visible"]),
                "infrared_url": self.img_url(first["infrared"]),
                "depth_url": self.img_url(first["depth"]),
                "queries": queries, "current_qid": current_qid,
                "owner": owner, "can_edit": owner == worker,
                "remaining": self.collab.remaining(image) if owner == worker else None}

    def bootstrap(self, worker: str) -> dict:
        image = self.collab.claimed_image(worker)
        return {"stats": self.collab.merged_stats(),
                "first_image": self.groups[0][0] if self.groups else None,
                "images": [g[0] for g in self.groups],
                "group": self.group_payload(image, worker=worker) if image else None,
                "collab": {"image": image,
                           "remaining": self.collab.remaining(image) if image else None,
                           "can_back": self.collab.can_back(worker)}}

    def next_target(self, image: str, qid: str):
        idx = self.image_index[image]
        items = self.groups[idx][1]
        seen = False
        for q, _ in items:
            if q == qid:
                seen = True
                continue
            if seen and q not in self.cancelled \
                    and self.store.get(q).get("status") != "done":
                return {"image": image, "qid": q}
        return {"image": image, "finished": True}

    def next_undone(self, image: str, qid: str):
        idx = self.image_index[image]
        items = self.groups[idx][1]
        seen = False
        for q, _ in items:
            if q == qid:
                seen = True
                continue
            if seen and q not in self.cancelled \
                    and self.store.get(q).get("status") != "done":
                return {"image": image, "qid": q}
        for i in range(idx + 1, len(self.groups)):
            for q, _ in self.groups[i][1]:
                if q not in self.cancelled \
                        and self.store.get(q).get("status") != "done":
                    return {"image": self.groups[i][0], "qid": q}
        return None

    def save_annotation(self, body: dict, worker: str) -> dict:
        qid = body.get("qid")
        if qid not in self.queries:
            return {"error": f"unknown qid: {qid}"}
        image = self.qid_image[qid]
        if not self.collab.can_edit(image, worker):
            return {"error": "该图未领取或由他人标注中，无法保存",
                    "forbidden": True}
        if body.get("skip"):
            entry = {"bbox": None, "status": "skipped"}
        else:
            bbox = normalize_bbox(body.get("bbox"))
            if bbox is None:
                return {"error": "invalid bbox"}
            entry = {"bbox": bbox, "status": "done"}
        self.collab.save_tmp(worker, image, qid, entry)
        return {"stats": self.collab.merged_stats(),
                "next": self.next_target(image, qid),
                "remaining": self.collab.remaining(image)}

    def serve_image(self, rel: str):
        rel = rel.replace("\\", "/")
        full = os.path.abspath(os.path.join(self.data_root, rel))
        try:
            common = os.path.commonpath([full, self.root_abs])
        except ValueError:
            return None
        if common != self.root_abs or not os.path.isfile(full):
            return None
        if os.path.basename(os.path.dirname(rel)) == "depth":
            return depth_jpeg(full), "image/jpeg"
        ctype = "image/jpeg" if os.path.splitext(rel)[1].lower() in (".jpg", ".jpeg") else "image/png"
        with open(full, "rb") as f:
            return f.read(), ctype

    def export(self) -> dict:
        view = self.collab.merged_view()
        effective = [qid for qid in self.all_qids if qid not in self.cancelled]
        answer = {}
        answer_full = {}
        for qid in sorted(effective):
            entry = view.get(qid)
            if not entry or entry.get("status") != "done":
                continue
            bbox = entry.get("bbox")
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4
                    and all(isinstance(v, (int, float)) and 0.0 <= v <= 1.0
                            for v in bbox)
                    and bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                continue
            answer[qid] = {"bbox": list(bbox)}
            item = self.queries[qid]
            answer_full[qid] = {"visible": item["visible"],
                                "infrared": item["infrared"],
                                "depth": item["depth"],
                                "query": item["query"],
                                "bbox": list(bbox)}
        missing = len(effective) - len(answer_full)
        if missing > 0:
            return {"error": f"查询数量不完整：共 {len(effective)} 条查询"
                             f"（已排除取消标注 {len(self.cancelled)} 条），"
                             f"已完成有效标注 {len(answer_full)} 条，还差 {missing} 条"
                             "（未标注/跳过/无效 bbox 均视为未完成），全部完成后再导出"}
        answer_path = os.path.join(self.out_dir, "answer.json")
        full_path = os.path.join(self.out_dir, "answer_full.json")
        zip_path = os.path.join(self.out_dir, "answer_full.zip")
        write_json(answer, answer_path)
        write_json(answer_full, full_path)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(full_path, "answer_full.json")
        stats = self.collab.merged_stats()
        return {"answer_file": answer_path, "answer_full_file": full_path,
                "zip_file": zip_path, "count": len(answer),
                "cancelled": len(self.cancelled), "stats": stats}

    def score_against(self, predictions_path: str,
                      answers_path: str | None = None,
                      threshold: float = 0.5) -> dict:
        answers_path = answers_path or os.path.join(self.out_dir, "answer.json")
        return score_files(predictions_path, answers_path, self.out_dir, threshold)


def make_handler(app: App):
    class AnnotateHandler(BaseHTTPRequestHandler):
        def _send(self, payload: bytes, content_type: str, status: int = 200,
                  extra_headers: list | None = None) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                for key, value in (extra_headers or []):
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass

        def _json(self, obj: dict, status: int = 200,
                  extra_headers: list | None = None) -> None:
            payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            headers = list(extra_headers or []) + [("Cache-Control", "no-store")]
            self._send(payload, "application/json; charset=utf-8", status, headers)

        def _serve_bytes(self, data: bytes, content_type: str,
                         extra_headers: list | None = None) -> None:
            self._send(data, content_type, 200, extra_headers)

        def _client_ip(self) -> str:
            return self.client_address[0] if self.client_address else ""

        def _is_admin(self) -> bool:
            return self._client_ip() == "127.0.0.1"

        def _worker(self) -> str:
            name = parse_cookie(self.headers.get("Cookie"), "annotator")
            session = app.collab.ensure_session(name)
            return session["name"]

        def _session(self) -> dict:
            name = parse_cookie(self.headers.get("Cookie"), "annotator")
            session = app.collab.ensure_session(name)
            return session

        def _cookie_header(self, name: str) -> tuple[str, str]:
            return ("Set-Cookie",
                    f"annotator={quote(name)}; Path=/; Max-Age={COOKIE_MAX_AGE}")

        def _origin_ok(self) -> bool:
            origin = self.headers.get("Origin")
            if not origin:
                return True
            try:
                return urlparse(origin).netloc == self.headers.get("Host", "")
            except ValueError:
                return False

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if not self._origin_ok():
                self._json({"error": "forbidden origin"}, 403)
                return
            if parsed.path == "/":
                try:
                    self._serve_bytes(read_bytes(pick_ui_path(parse_qs(parsed.query),
                                                              self.headers.get("User-Agent") or "")),
                                      "text/html; charset=utf-8", [("Cache-Control", "no-store")])
                except OSError:
                    self._serve_bytes(read_bytes(UI_PATH), "text/html; charset=utf-8",
                                      [("Cache-Control", "no-store")])
            elif parsed.path == "/roster.html":
                try:
                    self._serve_bytes(read_bytes(ROSTER_PATH), "text/html; charset=utf-8",
                                      [("Cache-Control", "no-store")])
                except OSError:
                    self._serve_bytes(b"<html><body><p>roster.html not found</p></body></html>",
                                      "text/html; charset=utf-8")
            elif parsed.path == "/api/session":
                session = self._session()
                self._json(session, extra_headers=[self._cookie_header(session["name"])])
            elif parsed.path == "/api/bootstrap":
                session = self._session()
                payload = app.bootstrap(session["name"])
                payload["session"] = {"name": session["name"],
                                      "is_admin": session["is_admin"],
                                      "display_name": session.get("display_name") or ""}
                self._json(payload, extra_headers=[self._cookie_header(session["name"])])
            elif parsed.path == "/api/roster":
                self._json(app.collab.roster())
            elif parsed.path == "/api/group":
                image = parse_qs(parsed.query).get("image", [None])[0]
                if not image:
                    self._json({"error": "missing image"}, 400)
                    return
                worker = self._worker()
                payload = app.group_payload(image, worker=worker)
                self._json(payload if payload is not None else {"error": "image not found"}, 404 if payload is None else 200)
            elif parsed.path == "/api/export":
                if not self._is_admin():
                    self._json({"error": "仅管理员（本机访问）可使用此功能"}, 403)
                    return
                self._json(app.export())
            elif parsed.path == "/api/imported":
                if not self._is_admin():
                    self._json({"error": "仅管理员（本机访问）可使用此功能"}, 403)
                    return
                self._json(app.imported_payload())
            elif parsed.path == "/api/jump":
                qs = parse_qs(parsed.query)
                image, qid = qs.get("image", [None])[0], qs.get("qid", [None])[0]
                if not image or not qid:
                    self._json({"error": "missing image/qid"}, 400)
                    return
                result = app.next_undone(image, qid)
                self._json(result if result is not None else {"none": True})
            elif parsed.path == "/img":
                rel = parse_qs(parsed.query).get("path", [None])[0]
                if not rel:
                    self._json({"error": "missing path"}, 400)
                    return
                result = app.serve_image(rel)
                if result is None:
                    self._json({"error": "not found"}, 404)
                else:
                    data, ctype = result
                    self._serve_bytes(data, ctype)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if not self._origin_ok():
                self._json({"error": "forbidden origin"}, 403)
                return
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY_BYTES:
                self._json({"error": "body too large"}, 413)
                return
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if parsed.path == "/api/score":
                if not self._is_admin():
                    self._json({"error": "仅管理员（本机访问）可使用此功能"}, 403)
                    return
                if body.get("predictions_b64"):
                    data = load_import_content(body["predictions_b64"], bool(body.get("is_zip")))
                    if not isinstance(data, dict):
                        self._json({"error": "提交结果解析失败（需 JSON 或 zip 内含 submission.json）"}, 400)
                        return
                    result = score_from_dict(data, os.path.join(app.out_dir, "answer.json"),
                                             app.out_dir, 0.5, "上传提交结果")
                else:
                    candidates = find_candidate_predictions(app.out_dir)
                    path = body.get("predictions") or (candidates[0] if candidates else None)
                    if not path:
                        result = {"error": "未找到提交结果文件", "candidates": candidates}
                    else:
                        result = app.score_against(path)
                self._json(result, 400 if "error" in result else 200)
                return
            if parsed.path == "/api/import":
                if not self._is_admin():
                    self._json({"error": "仅管理员（本机访问）可使用此功能"}, 403)
                    return
                result = app.import_predictions(body)
                self._json(result, 400 if "error" in result else 200)
                return
            if parsed.path == "/api/import_clear":
                if not self._is_admin():
                    self._json({"error": "仅管理员（本机访问）可使用此功能"}, 403)
                    return
                self._json(app.clear_imported())
                return
            worker = self._worker()
            if parsed.path == "/api/claim":
                self._json(app.collab.claim(worker, body.get("image")))
                return
            if parsed.path == "/api/release":
                self._json(app.collab.release(worker))
                return
            if parsed.path == "/api/claim_back":
                self._json(app.collab.claim_back(worker))
                return
            if parsed.path == "/api/rename_session":
                target = str(body.get("worker") or worker)
                if target != worker and not self._is_admin():
                    self._json({"ok": False, "error": "仅管理员可修改他人名称"}, 403)
                    return
                result = app.collab.set_display_name(target, str(body.get("name") or ""))
                self._json(result, 400 if not result.get("ok") else 200)
                return
            if parsed.path == "/api/heartbeat":
                self._json(app.collab.heartbeat(worker))
                return
            if parsed.path == "/api/admin_release":
                target = body.get("worker")
                if not self._is_admin():
                    self._json({"error": "仅管理员（本机访问）可强制释放"}, 403)
                    return
                if not target:
                    self._json({"error": "missing worker"}, 400)
                    return
                self._json(app.collab.admin_release(str(target)))
                return
            if parsed.path == "/api/admin_delete_session":
                target = body.get("worker")
                if not self._is_admin():
                    self._json({"deleted": False, "error": "仅管理员（本机访问）可删除会话"}, 403)
                    return
                if not target:
                    self._json({"deleted": False, "error": "missing worker"}, 400)
                    return
                self._json(app.collab.delete_session(self._worker(), str(target)))
                return
            if parsed.path == "/api/admin_set_admin":
                target = body.get("worker")
                if not self._is_admin():
                    self._json({"ok": False, "error": "仅管理员（本机访问）可修改权限"}, 403)
                    return
                if not target:
                    self._json({"ok": False, "error": "missing worker"}, 400)
                    return
                self._json(app.collab.set_admin(str(target), bool(body.get("is_admin"))))
                return
            if parsed.path != "/api/save":
                self._json({"error": "not found"}, 404)
                return
            result = app.save_annotation(body, worker)
            self._json(result, 400 if "error" in result else 200)

        def log_message(self, fmt, *args) -> None:
            pass

    return AnnotateHandler


def acquire_lock(lock_path: str):
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    f = open(lock_path, "a+")
    try:
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        f.close()
        return None
    f.seek(0)
    f.truncate()
    f.write(str(os.getpid()))
    f.flush()
    return f


def _lan_urls(port: int) -> list:
    try:
        ips = socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        return []
    return [f"http://{ip}:{port}/" for ip in ips if ip and not ip.startswith("127.")]


def run_annotate(args) -> None:
    app = App(args.data_root, args.queries_file, args.out_dir, args.max_images)

    lock = acquire_lock(os.path.join(app.out_dir, ".lock"))
    if lock is None:
        print(f"输出目录 {app.out_dir} 已有标注服务在运行，请勿重复启动")
        print("请先关闭已有服务，或使用 --out-dir 指定其他目录")
        sys.exit(1)
    try:
        server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(app))
    except OSError:
        lock.close()
        print(f"端口 {args.port} 已被占用：可能已有标注服务在运行，或上次未正常退出")
        print("请先关闭已有服务（或改用 --port 指定其他端口）再启动")
        sys.exit(1)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{args.port}/"
    initial = app.stats()
    if getattr(app.collab, "recovered", 0):
        print(f"启动恢复：已合并 {app.collab.recovered} 张崩溃残留认领图的临时标注（标注保留、图已回池）")
    for name in sorted(os.listdir(app.out_dir)):
        if name.startswith("tmp_") and name.endswith(".json"):
            print(f"警告：存在未归属的临时标注文件 {name}（未自动合并，请人工确认）")
    print(f"标注服务已启动: {url}")
    for lan in _lan_urls(args.port):
        print(f"局域网访问（供其他标注员）: {lan}")
    print(f"认领名单页: {url}roster.html")
    print(f"标注数据目录: {app.out_dir}")
    print(f"范围内查询 {initial['total']} 条，待标注 {initial['pending']} 条")
    print("本机访问为管理员，可强制释放；同一会话同时在手图片上限 1 张")
    print("按 Ctrl+C 停止（自动合并全部临时标注后退出）")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止服务并合并临时标注...")
        result = app.collab.shutdown()
        print(f"已合并 {result['merged']} 张在手图片的临时标注")
        print("服务已停止")
    finally:
        server.server_close()
        lock.close()


def latest_predictions(out_dir: str) -> str | None:
    roots = [out_dir, os.path.join(PROJECT_ROOT, "outputs")]
    candidates = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if re.match(r"^(predictions|submission)(_.+)?\.json$", name):
                candidates.append(os.path.join(root, name))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def run_gen_machine(args) -> None:
    out_dir = resolve_path(args.out_dir)
    preds_path = args.predictions or latest_predictions(out_dir)
    if not preds_path:
        print("未找到预测文件，请用 --predictions 指定")
        sys.exit(1)
    with open(preds_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        print(f"预测文件格式不正确（应为 {{qid: bbox}} 或 {{qid: {{bbox}}}}）: {preds_path}")
        sys.exit(1)
    data = normalize_imported(raw)
    if not data:
        print("预测文件中没有有效 bbox")
        sys.exit(1)
    if args.skip_done:
        store = Store(os.path.join(out_dir, "annotations.json"))
        done = {qid for qid in data
                if store.get(qid).get("status") in ("done", "skipped")}
        data = {qid: bbox for qid, bbox in data.items() if qid not in done}
    write_json({"name": os.path.basename(preds_path), "data": data},
               os.path.join(out_dir, "imported_annotations.json"))
    print(f"来源: {preds_path}")
    print(f"生成机器标注: {len(data)} 条 → {os.path.join(out_dir, 'imported_annotations.json')}")
    print("启动标注服务后自动加载，界面以蓝色框显示机器标注，可与人工标注对比检查")


def run_score(args) -> None:
    out_dir = resolve_path(args.out_dir)
    answers_path = args.answers or os.path.join(out_dir, "answer.json")
    result = score_files(args.predictions, answers_path, out_dir, args.threshold)
    if "error" in result:
        print(result["error"])
        sys.exit(1)
    report = result["report"]
    print(f"提交文件: {report['predictions']}")
    print(f"答案文件: {report['answers']}")
    print(f"样本数: {report['total']}")
    print(f"ACC@{report['threshold']} = {report['acc']:.4f} ({report['correct']}/{report['total']})")
    print(f"Mean IoU = {report['mean_iou']:.4f}")
    errors = result["errors_head"]
    if errors:
        print(f"错误样本（前 {len(errors)} 条，IoU 升序）:")
        for e in errors:
            print(f"  {e['qid']}  iou={e['iou']:.4f}  pred={e['pred']}  gt={e['gt']}")
    print(f"报告已保存: {result['report_file']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="人工标注工具：浏览器拖拽画框标注真值 bbox，并与提交结果比对生成分数")
    ap.add_argument("--data-root", default="dataset", help="数据集根目录（默认 dataset）")
    ap.add_argument("--queries-file", default="queries/queries.json", help="查询 JSON 相对路径")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="标注、答案与分数报告输出目录")
    ap.add_argument("--port", type=int, default=8765, help="本地服务端口")
    ap.add_argument("--max-images", type=int, default=None, help="仅标注前 N 张图")
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    sub = ap.add_subparsers(dest="command")
    score_p = sub.add_parser("score", help="比对提交结果与人工标注，生成分数")
    score_p.add_argument("--predictions", required=True, help="提交结果 JSON 路径（预测 bbox）")
    score_p.add_argument("--answers", default=None, help="人工标注答案 JSON（默认 <out-dir>/answer.json）")
    score_p.add_argument("--threshold", type=float, default=0.5, help="IoU 阈值，默认 0.5")
    gen_p = sub.add_parser("gen-machine", help="从模型预测文件生成机器标注，供界面导入对比检查")
    gen_p.add_argument("--predictions", default=None, help="预测 JSON 路径（默认自动找 outputs 下最新文件）")
    gen_p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="标注输出目录")
    gen_p.add_argument("--skip-done", action="store_true", help="跳过已人工标注/跳过的 qid")
    args = ap.parse_args()

    if args.command == "score":
        run_score(args)
    elif args.command == "gen-machine":
        run_gen_machine(args)
    else:
        run_annotate(args)


if __name__ == "__main__":
    main()

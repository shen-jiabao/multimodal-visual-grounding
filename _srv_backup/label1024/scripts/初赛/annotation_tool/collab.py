from __future__ import annotations

import json
import os
import random
import re
import string
import threading
import time

TIMEOUT_SECONDS = 600
HEARTBEAT_INTERVAL = 60
COOKIE_MAX_AGE = 31536000


def write_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


class Collab:
    def __init__(self, out_dir: str, store, groups: list, all_qids: list,
                 cancelled: set | None = None):
        self.out_dir = out_dir
        self.store = store
        self.groups = groups
        self.all_qids = all_qids
        self.cancelled = cancelled or set()
        self.lock = threading.Lock()
        self.sessions_path = os.path.join(out_dir, "sessions.json")
        self.assignment_path = os.path.join(out_dir, "assignment.json")
        self.log_path = os.path.join(out_dir, "release_log.json")
        self.sessions = load_json(self.sessions_path, {})
        self.assignment = load_json(self.assignment_path, {})
        self.log = load_json(self.log_path, [])
        if not isinstance(self.sessions, dict):
            self.sessions = {}
        if not isinstance(self.assignment, dict):
            self.assignment = {}
        if not isinstance(self.log, list):
            self.log = []
        self._migrate_counts()
        self.recovered = self._recover_orphans()

    def _recover_orphans(self) -> int:
        with self.lock:
            pending = [(image, rec["worker"])
                       for image, rec in list(self.assignment.items())
                       if rec.get("status") == "active"]
            for image, worker in pending:
                self.release_image_locked(image, worker, "recover")
        return len(pending)

    def _migrate_counts(self) -> None:
        migrated = False
        for w, rec in self.sessions.items():
            if "done_count" in rec:
                continue
            rec["done_count"] = sum(int(l.get("merged", 0) or 0)
                                    for l in self.log if l.get("worker") == w)
            migrated = True
        if migrated:
            self.save_sessions()

    def save_sessions(self) -> None:
        write_json(self.sessions, self.sessions_path)

    def save_assignment(self) -> None:
        write_json(self.assignment, self.assignment_path)

    def save_log(self) -> None:
        self.log = self.log[-200:]
        write_json(self.log, self.log_path)

    def tmp_path(self, worker: str) -> str:
        safe = re.sub(r"[^\w\-]", "_", worker)
        return os.path.join(self.out_dir, f"tmp_{safe}.json")

    def ensure_session(self, name: str) -> dict:
        with self.lock:
            if not name:
                name = self.new_name()
            rec = self.sessions.get(name)
            if rec is None:
                rec = {"is_admin": False, "created_at": time.time()}
                self.sessions[name] = rec
            rec["last_seen"] = time.time()
            self.save_sessions()
            return {"name": name, "is_admin": bool(rec["is_admin"]),
                    "display_name": rec.get("display_name") or ""}

    def set_display_name(self, worker: str, name: str) -> dict:
        with self.lock:
            rec = self.sessions.get(worker)
            if rec is None:
                return {"ok": False, "error": "会话不存在"}
            name = (name or "").strip()
            if not name:
                return {"ok": False, "error": "名称不能为空"}
            if len(name) > 20:
                return {"ok": False, "error": "名称过长（最多 20 字符）"}
            for other, other_rec in self.sessions.items():
                if other == worker:
                    continue
                if name == other or name == (other_rec.get("display_name") or ""):
                    return {"ok": False, "error": "名称已被其他会话使用"}
            rec["display_name"] = name
            self.save_sessions()
            return {"ok": True, "display_name": name}

    def set_admin(self, worker: str, is_admin: bool) -> dict:
        with self.lock:
            rec = self.sessions.get(worker)
            if rec is None:
                return {"ok": False, "error": "会话不存在"}
            rec["is_admin"] = bool(is_admin)
            self.save_sessions()
            return {"ok": True, "is_admin": bool(is_admin)}

    def new_name(self) -> str:
        while True:
            name = "标注员-" + "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
            if name not in self.sessions:
                return name

    def image_owner(self, image: str) -> str | None:
        rec = self.assignment.get(image)
        if rec and rec.get("status") == "active":
            return rec["worker"]
        return None

    def can_edit(self, image: str, worker: str) -> bool:
        return self.image_owner(image) == worker

    def claimed_image(self, worker: str) -> str | None:
        for image, rec in self.assignment.items():
            if rec.get("status") == "active" and rec["worker"] == worker:
                return image
        return None

    def remaining(self, image: str) -> int | None:
        rec = self.assignment.get(image)
        if not rec or rec.get("status") != "active":
            return None
        return max(0, int(TIMEOUT_SECONDS - (time.time() - rec["last_active"])))

    def claim(self, worker: str, image: str | None = None) -> dict:
        self.cleanup_timeouts()
        with self.lock:
            mine = self.claimed_image_locked(worker)
            if mine:
                self.touch_locked(mine)
                return {"claimed": True, "image": mine, "already": True,
                        "remaining": self.remaining(mine)}
            if image is not None:
                if image not in [g[0] for g in self.groups]:
                    return {"claimed": False, "error": "图片不存在"}
                rec = self.assignment.get(image)
                if rec and rec.get("status") == "active":
                    return {"claimed": False,
                            "error": f"该图已由 {rec['worker']} 标注中"}
                self._assign_locked(worker, image)
                return {"claimed": True, "image": image,
                        "remaining": TIMEOUT_SECONDS,
                        "can_back": self.can_back(worker)}
            view = self.merged_view()
            candidates = [g[0] for g in self.groups
                          if g[0] not in self.assignment
                          and any(not view.get(q[0])
                                  or view[q[0]].get("status") in ("pending", "skipped")
                                  for q in g[1])]
            if not candidates:
                return {"claimed": False, "error": "暂无未标注且未认领的图片"}
            image = random.choice(candidates)
            self._assign_locked(worker, image)
            return {"claimed": True, "image": image,
                    "remaining": TIMEOUT_SECONDS,
                    "can_back": self.can_back(worker)}

    def _assign_locked(self, worker: str, image: str) -> None:
        now = time.time()
        self.assignment[image] = {"worker": worker, "claimed_at": now,
                                  "last_active": now, "status": "active"}
        self.save_assignment()
        self.push_claim_stack_locked(worker, image)

    def push_claim_stack_locked(self, worker: str, image: str) -> None:
        rec = self.sessions.setdefault(worker, {})
        stack = rec.get("claim_stack")
        if not isinstance(stack, list):
            stack = []
            rec["claim_stack"] = stack
        if image in stack:
            stack.remove(image)
        stack.append(image)
        self.save_sessions()

    def can_back(self, worker: str) -> bool:
        rec = self.sessions.get(worker)
        stack = rec.get("claim_stack") if rec else None
        if not isinstance(stack, list) or not stack:
            return False
        if self.claimed_image(worker):
            return len(stack) >= 2
        return True

    def claim_back(self, worker: str) -> dict:
        self.cleanup_timeouts()
        with self.lock:
            rec = self.sessions.get(worker)
            stack = rec.get("claim_stack") if rec else None
            if not isinstance(stack, list) or not stack:
                return {"backed": False, "error": "没有认领历史"}
            mine = self.claimed_image_locked(worker)
            if mine:
                if len(stack) == 1:
                    return {"backed": False, "error": "没有更早的认领记录"}
                target = stack[-2]
            else:
                target = stack[-1]
            owner = self.image_owner(target)
            if owner and owner != worker:
                return {"backed": False,
                        "error": f"目标图片 {target} 已被 {owner} 认领"}
            if mine:
                self.release_image_locked(mine, worker, "release")
                stack.pop()
            if target in stack:
                stack.remove(target)
            stack.append(target)
            self.save_sessions()
            now = time.time()
            self.assignment[target] = {"worker": worker, "claimed_at": now,
                                       "last_active": now, "status": "active"}
            self.save_assignment()
            return {"backed": True, "image": target,
                    "can_back": self.can_back(worker)}

    def release(self, worker: str) -> dict:
        with self.lock:
            image = self.claimed_image_locked(worker)
            if image is None:
                return {"released": False, "error": "没有在手图片"}
            self.release_image_locked(image, worker, "release")
            return {"released": True, "image": image}

    def admin_release(self, worker: str) -> dict:
        with self.lock:
            image = self.claimed_image_locked(worker)
            if image is None:
                return {"released": False, "error": f"会话 {worker} 无在手图片"}
            self.release_image_locked(image, worker, "admin")
            return {"released": True, "image": image, "worker": worker}

    def delete_session(self, admin: str, worker: str) -> dict:
        self.cleanup_timeouts()
        with self.lock:
            if worker not in self.sessions:
                return {"deleted": False, "error": f"会话 {worker} 不存在"}
            if worker == admin:
                return {"deleted": False, "error": "不能删除当前管理员会话"}
            image = self.claimed_image_locked(worker)
            if image:
                self.release_image_locked(image, worker, "admin")
            self.sessions.pop(worker, None)
            self.save_sessions()
            tmp = self.tmp_path(worker)
            if os.path.exists(tmp):
                os.remove(tmp)
            return {"deleted": True, "worker": worker, "image": image}

    def heartbeat(self, worker: str) -> dict:
        self.cleanup_timeouts()
        with self.lock:
            mine = self.claimed_image_locked(worker)
            if mine:
                self.touch_locked(mine)
            return {"image": mine}

    def claimed_image_locked(self, worker: str) -> str | None:
        for image, rec in self.assignment.items():
            if rec.get("status") == "active" and rec["worker"] == worker:
                return image
        return None

    def touch_locked(self, image: str) -> None:
        rec = self.assignment.get(image)
        if rec:
            rec["last_active"] = time.time()
            self.save_assignment()

    def _merge_tmp_locked(self, worker: str, image: str) -> int:
        tmp = self.tmp_path(worker)
        if not os.path.exists(tmp):
            return 0
        rec = load_json(tmp, {})
        data = rec.get("data") if isinstance(rec, dict) else None
        merged = 0
        if isinstance(data, dict) and rec.get("image") == image:
            entries = {qid: {"bbox": entry.get("bbox"),
                             "status": entry.get("status", "pending")}
                       for qid, entry in data.items()
                       if isinstance(entry, dict) and qid in self.all_qids}
            if entries:
                self.store.update_many(entries)
                merged = len(entries)
                valid = sum(1 for e in entries.values()
                            if e.get("status") == "done"
                            and isinstance(e.get("bbox"), list)
                            and len(e["bbox"]) == 4)
                session = self.sessions.get(worker)
                if session:
                    session["done_count"] = session.get("done_count", 0) + valid
                    self.save_sessions()
        os.remove(tmp)
        return merged

    def release_image_locked(self, image: str, worker: str, reason: str) -> None:
        merged = self._merge_tmp_locked(worker, image)
        self.assignment.pop(image, None)
        self.save_assignment()
        self.log.append({"image": image, "worker": worker, "reason": reason,
                         "merged": merged, "at": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.save_log()

    def save_tmp(self, worker: str, image: str, qid: str, entry: dict) -> None:
        with self.lock:
            path = self.tmp_path(worker)
            rec = load_json(path, {})
            if not isinstance(rec, dict) or rec.get("image") != image \
                    or not isinstance(rec.get("data"), dict):
                rec = {"image": image, "data": {}}
            rec["data"][qid] = entry
            write_json(rec, path)
            a = self.assignment.get(image)
            if a and a["worker"] == worker and a.get("status") == "active":
                a["last_active"] = time.time()
                self.save_assignment()

    def merged_view(self) -> dict:
        view = {qid: dict(entry) for qid, entry in self.store.snapshot().items()
                if entry and qid in self.all_qids}
        for name in os.listdir(self.out_dir):
            if not name.startswith("tmp_") or not name.endswith(".json"):
                continue
            rec = load_json(os.path.join(self.out_dir, name), {})
            data = rec.get("data") if isinstance(rec, dict) else None
            if not isinstance(data, dict):
                continue
            for qid, entry in data.items():
                if isinstance(entry, dict) and qid in self.all_qids:
                    view[qid] = {"bbox": entry.get("bbox"),
                                 "status": entry.get("status", "pending")}
        return view

    def merged_stats(self) -> dict:
        view = self.merged_view()
        done = skipped = 0
        for qid in self.all_qids:
            if qid in self.cancelled:
                continue
            status = view.get(qid, {}).get("status")
            if status == "done":
                done += 1
            elif status == "skipped":
                skipped += 1
        total = len(self.all_qids) - len(self.cancelled)
        return {"total": total, "done": done, "skipped": skipped,
                "pending": total - done - skipped,
                "cancelled": len(self.cancelled)}

    def cleanup_timeouts(self) -> None:
        now = time.time()
        with self.lock:
            for image, rec in list(self.assignment.items()):
                if rec.get("status") == "active" \
                        and now - rec.get("last_active", 0) > TIMEOUT_SECONDS:
                    self.release_image_locked(image, rec["worker"], "timeout")

    def roster(self) -> dict:
        self.cleanup_timeouts()
        now = time.time()
        rows = []
        for name, rec in self.sessions.items():
            image = self.claimed_image(name)
            remaining = self.remaining(image) if image else None
            claimed_at = None
            if image:
                a = self.assignment[image]
                claimed_at = a.get("claimed_at")
            rows.append({"name": name, "display_name": rec.get("display_name") or "",
                         "is_admin": bool(rec.get("is_admin")),
                         "image": image, "claimed_at": claimed_at,
                         "remaining": remaining,
                         "done_count": rec.get("done_count", 0),
                         "status": "active" if image else "idle"})
        rows.sort(key=lambda r: (not r["is_admin"], r["display_name"] or r["name"]))
        unclaimed = sum(1 for image, _ in self.groups if image not in self.assignment)
        return {"rows": rows, "stats": self.merged_stats(),
                "unclaimed_images": unclaimed, "log": self.log[-20:],
                "now": now}

    def shutdown(self) -> dict:
        self.cleanup_timeouts()
        with self.lock:
            pending = [(image, rec["worker"]) for image, rec in list(self.assignment.items())
                       if rec.get("status") == "active"]
            for image, worker in pending:
                self.release_image_locked(image, worker, "shutdown")
            return {"merged": len(pending)}

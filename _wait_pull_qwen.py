# -*- coding: utf-8 -*-
"""Qwen3-VL 全量等待器 + 自动拉回。

- 轮询远端 `qwen_full.log`：出现收尾标记「完成 direct」且进程退出 → 判定完成
- 完成后经 SFTP 拉回两份 JSON 到本地，并做结构校验（键集/条数/命中率/SoM 原因分布）
- 全程捕获 SSH 异常自动重连（沿用本项目 exec_retry 约定）
- `--once` 只检查一次；已完成则拉回，未完成则退出码 2（供人工/定时调用）

用法：
    设好 SSH_HOST/SSH_PORT/SSH_USER/SSH_PW 后运行
    python _wait_pull_qwen.py [--once] [--interval 300]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import paramiko

R = Path(r"d:\Desktop\2026AiC")
REMOTE_OUT = "/srv/output"
LOG = f"{REMOTE_OUT}/qwen_full.log"
FILES = ["qwen3vl_all.json", "qwen3vl_som_all.json"]
# 必须锚定行首：收尾行是 "完成 direct 9555(有框N) som 2014(有框N)"，
# 而启动行 "已完成 direct=200 som=4" 含有子串 "完成 direct"，
# 不加 ^ 会导致一启动就误判完成。
DONE_GREP = r"^完成 direct "

N_ENTRIES = 9555
N_ORDINAL = 2014
# 日志超过该秒数未更新即视为停滞（正常每张图都会刷一行，远小于此值）
STALE_SEC = 1800
TEST_JSON = R / "初赛数据集-基于大模型的多模态视觉理解与推理" / "test.json"
LOG_PATH = R / "_wait_pull_qwen.log"


def log(msg: str) -> None:
    """同时写控制台与日志文件并立即 flush。

    不使用 PowerShell 的 Tee-Object：它会在管道里缓冲，导致等待器运行期间
    完全不可观测（实测 5 分钟后日志仍只有首行）。
    """
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def sha256(p: Path) -> str:
    """计算文件 SHA256（大写十六进制）。"""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


class Remote:
    """带自动重连的远端会话封装。"""

    def __init__(self) -> None:
        self.cli: Optional[paramiko.SSHClient] = None

    def _connect(self) -> None:
        cli = paramiko.SSHClient()
        cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        cli.connect(os.environ["SSH_HOST"], port=int(os.environ["SSH_PORT"]),
                    username=os.environ["SSH_USER"],
                    password=os.environ["SSH_PW"], timeout=25,
                    banner_timeout=30, auth_timeout=30)
        self.cli = cli

    def run(self, cmd: str, timeout: int = 90) -> str:
        """执行远端命令；遇 SSH 瞬断自动重连重试（最多 5 次）。"""
        last = None
        for attempt in range(5):
            try:
                if self.cli is None:
                    self._connect()
                _, out, _ = self.cli.exec_command(cmd, timeout=timeout)
                return out.read().decode("utf-8", "replace")
            except (paramiko.SSHException, OSError) as ex:
                last = ex
                log(f"[warn] SSH 重连 {attempt + 1}/5: {ex!r}")
                try:
                    if self.cli:
                        self.cli.close()
                except Exception:
                    pass
                self.cli = None
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"远端命令连续失败: {last!r}")

    def pull(self, name: str) -> Path:
        """下载远端产物到本地同名文件；遇瞬断重连重试。"""
        dst = R / name
        last = None
        for attempt in range(5):
            try:
                if self.cli is None:
                    self._connect()
                sftp = self.cli.open_sftp()
                sftp.get(f"{REMOTE_OUT}/{name}", str(dst))
                sftp.close()
                return dst
            except (paramiko.SSHException, OSError, IOError) as ex:
                last = ex
                log(f"[warn] 下载重试 {attempt + 1}/5 {name}: {ex!r}",
                      flush=True)
                try:
                    if self.cli:
                        self.cli.close()
                except Exception:
                    pass
                self.cli = None
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"下载失败 {name}: {last!r}")

    def close(self) -> None:
        if self.cli:
            self.cli.close()
            self.cli = None


def check(rem: Remote) -> tuple[bool, str]:
    """判断是否完成，并给出日志新鲜度。

    完成判定只用收尾标记：远端脚本先落盘两份 JSON、再打印「完成 direct」，
    因此标记出现即代表产物已是终态。

    不使用 `pgrep -f` 判活：远端经 `sh -c` 执行时，包装进程自身的命令行里
    也含脚本名，会被 pgrep 匹配到（实测恒为 +1），导致「进程数为 0」永不成立。
    改用日志 mtime 做卡死检测（超过 STALE_SEC 未更新即告警）。
    """
    cmd = (f"grep -cE '{DONE_GREP}' {LOG}; "
           f"stat -c %Y {LOG}; date +%s; tail -n 1 {LOG}")
    lines = rem.run(cmd).strip().splitlines()
    marker = int(lines[0]) if lines[0].strip().isdigit() else 0
    mtime = int(lines[1])
    now = int(lines[2])
    age = now - mtime
    last = lines[3] if len(lines) > 3 else ""
    stale = age > STALE_SEC
    info = (f"日志静默 {age // 60}min{' [停滞!]' if stale else ''} | "
            f"{last}")
    return marker >= 1, info


def verify(paths: list[Path]) -> bool:
    """结构校验：键集与 test.json 一致、bbox 合法、统计各阶段命中。"""
    direct = json.loads((R / FILES[0]).read_text(encoding="utf-8"))
    som = json.loads((R / FILES[1]).read_text(encoding="utf-8"))
    test = json.loads(TEST_JSON.read_text(encoding="utf-8"))
    ok = True

    if set(direct) != set(test):
        log(f"[FAIL] direct 键集不一致: 缺 {len(set(test) - set(direct))} "
              f"多 {len(set(direct) - set(test))}")
        ok = False
    if len(som) != N_ORDINAL:
        log(f"[FAIL] som 条数 {len(som)} != 序数 {N_ORDINAL}")
        ok = False

    bad = []
    for k, v in direct.items():
        b = v.get("bbox")
        if b is None:
            continue
        if not (0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1):
            bad.append(k)
    if bad:
        log(f"[FAIL] direct bbox 非法 {len(bad)} 条 {bad[:5]}")
        ok = False

    hit = sum(1 for v in direct.values() if v.get("bbox"))
    miss = Counter(v.get("err", "no_box") for v in direct.values()
                   if not v.get("bbox"))
    log(f"direct {len(direct)} 命中 {hit} ({hit / len(direct):.1%}) "
          f"未命中原因 {dict(miss)}")
    s_hit = sum(1 for v in som.values() if v.get("bbox"))
    reasons = Counter(v.get("reason", "ok") for v in som.values())
    log(f"som    {len(som)} 有框 {s_hit} ({s_hit / max(len(som), 1):.1%}) "
          f"原因 {dict(reasons)}")
    n_enum = sum(1 for v in som.values() if v.get("n_cands") is not None
                 and v["n_cands"] > 1)
    log(f"som 枚举出多候选 {n_enum}/{len(som)}")

    for p in paths:
        log(f"SHA256 {p.name} {sha256(p)}  ({p.stat().st_size} B)")
    log("校验结果:", "PASS" if ok else "FAIL")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="只检查一次")
    ap.add_argument("--interval", type=int, default=300, help="轮询秒数")
    args = ap.parse_args()
    log(f"\n===== 等待器启动 {time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"interval={args.interval}s once={args.once} =====")

    rem = Remote()
    try:
        t0 = time.time()
        n = 0
        while True:
            n += 1
            try:
                done, info = check(rem)
            except Exception as ex:
                log(f"[{n}] 检查失败（将重试）: {ex!r}")
                done, info = False, "check_error"
            log(f"[{n}] {time.strftime('%H:%M:%S')} 已用 "
                  f"{(time.time() - t0) / 3600:.2f}h {info}")
            if done:
                break
            if args.once:
                log("未完成，退出（--once）")
                return 2
            time.sleep(args.interval)

        log("== 已判定完成，开始拉回 ==")
        paths = [rem.pull(f) for f in FILES]
        log("拉回完成:", [p.name for p in paths])
    finally:
        rem.close()

    return 0 if verify(paths) else 1


if __name__ == "__main__":
    raise SystemExit(main())

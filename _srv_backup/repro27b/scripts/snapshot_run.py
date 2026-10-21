"""归档快照脚本：记录一次推理产物对应的「环境参数 + 项目代码 + 配置」。

在云端（或本地）运行，向指定 run 目录写入三份快照，确保「结果 ↔ 环境 ↔ 代码」可复现：

    env_snapshot.json    主机 / OS / Python / torch / CUDA / GPU / 关键包版本 + nvidia-smi
    code_snapshot.json   git commit / 分支 / dirty 列表 + 源码目录 MD5 清单
    config_snapshot.yaml 生效配置副本

用法：
    python scripts/snapshot_run.py \
        --out-dir outputs/复赛/archive/run_20260923_base \
        --config configs/cloud_gdino.yaml --label "复赛-GDINO基线"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone

import yaml

import _bootstrap  # noqa: F401  (注入项目根到 sys.path)

from src.config import PROJECT_ROOT, load_config

# 需记录版本的关键依赖
KEY_PACKAGES = [
    "torch", "torchvision", "transformers", "tokenizers", "accelerate",
    "safetensors", "numpy", "Pillow", "opencv-python", "PyYAML", "pytest",
    "huggingface_hub",
]
# 纳入代码 MD5 清单的目录 / 文件（相对项目根）
MANIFEST_ROOTS = ["src", "scripts", "configs", "tests", "tools"]
MANIFEST_FILES = ["README.md", "requirements.txt"]
# 清单排除的前缀（备份 / 产物 / 缓存）
MANIFEST_EXCLUDE = ("tools/rollback_backup", "outputs", "data", "dataset",
                    "checkpoints", "__pycache__")


def _md5(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git(*args: str) -> str:
    try:
        r = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=30,
        )
        return r.stdout.strip()
    except Exception:
        return ""


def _nvidia_smi() -> str:
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or r.stderr.strip()
    except Exception as e:  # nvidia-smi 不存在（CPU 环境）
        return f"<unavailable: {e}>"


def _pkg_versions() -> dict:
    from importlib import metadata
    out = {}
    for name in KEY_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    # opencv 可能装的是 headless 变体
    if out.get("opencv-python") is None:
        try:
            out["opencv-python"] = metadata.version("opencv-python-headless")
        except metadata.PackageNotFoundError:
            pass
    return out


def _torch_info() -> dict:
    info = {"available": False}
    try:
        import torch
        info = {
            "available": True,
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
        }
        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    except Exception as e:
        info["error"] = repr(e)
    return info


def collect_env(label: str | None = None) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": os.sys.executable,
        "packages": _pkg_versions(),
        "torch": _torch_info(),
        "nvidia_smi": _nvidia_smi(),
    }


def _should_include(rel: str) -> bool:
    return not any(rel == p or rel.startswith(p + "/") for p in MANIFEST_EXCLUDE)


def collect_code() -> dict:
    manifest: dict = {}
    for root in MANIFEST_ROOTS:
        base = os.path.join(PROJECT_ROOT, root)
        if not os.path.isdir(base):
            continue
        for dp, dns, fns in os.walk(base):
            dns[:] = [d for d in dns if d != "__pycache__"]
            for fn in fns:
                if fn.endswith((".pyc", ".pyo")):
                    continue
                full = os.path.join(dp, fn)
                rel = os.path.relpath(full, PROJECT_ROOT).replace(os.sep, "/")
                if _should_include(rel):
                    manifest[rel] = _md5(full)
    for fn in MANIFEST_FILES:
        full = os.path.join(PROJECT_ROOT, fn)
        if os.path.isfile(full):
            manifest[fn] = _md5(full)
    info = {
        "git_head": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_describe": _git("describe", "--tags", "--always", "--dirty"),
        "git_dirty": [l for l in _git("status", "--porcelain").splitlines() if l.strip()],
        "manifest_files": len(manifest),
        "manifest": dict(sorted(manifest.items())),
    }
    # 无 .git（代码为上传而非 clone）时，回退读取上传附带的上游 commit 信息
    if not info["git_head"]:
        origin_path = os.path.join(PROJECT_ROOT, "_code_origin.json")
        if os.path.isfile(origin_path):
            try:
                with open(origin_path, encoding="utf-8") as f:
                    origin = json.load(f)
                for k in ("git_head", "git_branch", "git_dirty"):
                    if origin.get(k):
                        info[k] = origin[k]
                info["origin_source"] = "_code_origin.json"
            except Exception:
                pass
    return info


def _write_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="写入 run 目录的环境/代码/配置快照")
    p.add_argument("--out-dir", required=True, help="run 归档目录（不存在则创建）")
    p.add_argument("--config", default=None, help="配置文件（默认 configs/config.yaml）")
    p.add_argument("--data-root", default=None, help="覆盖 data.root")
    p.add_argument("--label", default=None, help="本次运行说明（写入 env_snapshot）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {"data.root": args.data_root} if args.data_root else {}
    cfg = load_config(args.config, overrides)

    os.makedirs(args.out_dir, exist_ok=True)
    env = collect_env(args.label)
    code = collect_code()
    _write_json(env, os.path.join(args.out_dir, "env_snapshot.json"))
    _write_json(code, os.path.join(args.out_dir, "code_snapshot.json"))
    with open(os.path.join(args.out_dir, "config_snapshot.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, allow_unicode=True, sort_keys=False)

    print(f"快照已写入：{args.out_dir}")
    print(f"  git   : {code['git_head']} ({code['git_branch']}) dirty={len(code['git_dirty'])}")
    print(f"  files : {code['manifest_files']} 个源文件已记 MD5")
    print(f"  python: {env['python_version']}  torch: {env['torch'].get('version')} "
          f"cuda: {env['torch'].get('cuda_available')}")


if __name__ == "__main__":
    main()

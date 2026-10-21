# -*- coding: utf-8 -*-
'''归档规范化：以标准归档为模板，把一次推理的产物目录补齐、改名、重算 MD5。

标准归档结构
------------
参考 outputs/复赛/archive/run_20260924_090232_qwen_multiimg/
（该目录内容与同名 .tar.gz 的解包内容一致）：

    predictions.json          全量预测结果
    submission.json           提交包内的 JSON
    submission(<成绩>).zip     提交 zip；拿到官方成绩后按成绩重命名
    config_snapshot.yaml      生效配置快照
    env_snapshot.json         环境快照
    code_snapshot.json        代码快照（含逐文件 MD5 清单）
    code.tar.gz               与 manifest 逐字节一致的源码包
    MD5SUMS.txt               上述 7 项（固定顺序）；run.log 不进清单
    run.log                   运行日志

    run_<ts>_<desc>.tar.gz    以上目录的打包回传件（内含同名顶层目录）

本脚本把任意一次运行的产物目录规范到该结构：

  1. 按 code_snapshot.json 的 manifest 从 --code-source 抽取源码重建
     code.tar.gz，并逐文件校验 MD5，保证 内容 == manifest；
  2. 把 submission.zip 按官方成绩重命名为 submission(<成绩>).zip；
  3. 按固定顺序重写 MD5SUMS.txt；
  4. 重新打包外层 run_<ts>_<desc>.tar.gz。

用法
----
    python scripts/复赛/normalize_archive.py \
        --run outputs/复赛/archive/run_20260928_003617_qwen_diag \
        --code-source aic_code_upload.tar.gz --score 0.7283

    # 只体检、不写盘
    python scripts/复赛/normalize_archive.py --run <dir> --code-source <src> --check

--code-source 接受 tar.gz 或目录；若省略且已存在 code.tar.gz，则只做校验与
改名/清单/打包三步。
'''
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import io
import json
import os
import re
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BS = chr(92)                      # 反斜杠，避免源码里出现转义
CODE_TAR = 'code.tar.gz'
SNAPSHOT = 'code_snapshot.json'
MD5SUMS = 'MD5SUMS.txt'
LOG = 'run.log'


# ------------------------------------------------------------------ 工具
def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _clean(name: str) -> str:
    name = name.replace(BS, '/')
    while name.startswith('./'):
        name = name[2:]
    return name


def run_mtime(run_dir: str) -> int:
    '''从 run_<ts>_<desc> 目录名解析时间戳，作为包内 mtime（可复现）。'''
    base = os.path.basename(run_dir.rstrip('/' + BS))
    m = re.search(r'run_(\d{8})_(\d{6})', base)
    if m:
        d = datetime.datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
        return int(d.timestamp())
    return int(os.path.getmtime(run_dir))


# ------------------------------------------------------------------ 读取
def load_manifest(run_dir: str) -> dict:
    path = os.path.join(run_dir, SNAPSHOT)
    with open(path, encoding='utf-8') as f:
        snap = json.load(f)
    manifest = snap.get('manifest') or {}
    if not manifest:
        raise SystemExit('[error] ' + path + ' 内没有 manifest')
    return manifest


def collect_from_source(src: str, want) -> dict:
    '''按 manifest 的路径清单从源码来源取字节。

    目录与 tar.gz 都支持；只取清单内的文件，因此不会遍历 .venv/.git/产物目录。
    '''
    out = {}
    if os.path.isdir(src):
        for rel in want:
            p = os.path.join(src, rel.replace('/', os.sep))
            if os.path.isfile(p):
                with open(p, 'rb') as f:
                    out[rel] = f.read()
        return out
    with tarfile.open(src) as t:
        index = {}
        for m in t.getmembers():
            if not m.isfile():
                continue
            name = _clean(m.name)
            index.setdefault(name, m)
            parts = name.split('/')
            if len(parts) > 1:            # 兼容带单层顶层目录的包
                index.setdefault('/'.join(parts[1:]), m)
        for rel in want:
            m = index.get(rel)
            if m is not None:
                out[rel] = t.extractfile(m).read()
    return out


def find_submission_zip(run_dir: str):
    '''返回目录内的提交 zip 名（优先已按成绩命名的那份）。'''
    cands = [n for n in sorted(os.listdir(run_dir))
             if n.startswith('submission') and n.endswith('.zip')]
    named = [n for n in cands if n != 'submission.zip']
    if named:
        return named[0]
    return cands[0] if cands else None


# ------------------------------------------------------------------ 写入
def build_code_tar(files: dict, out_path: str, mtime: int) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w', format=tarfile.GNU_FORMAT) as t:
        seen = set()
        for rel in sorted(files):
            parts = rel.split('/')
            for i in range(1, len(parts)):
                d = '/'.join(parts[:i])
                if d in seen:
                    continue
                seen.add(d)
                ti = tarfile.TarInfo(d + '/')
                ti.type = tarfile.DIRTYPE
                ti.mode = 0o755
                ti.mtime = mtime
                ti.uid = ti.gid = 0
                ti.uname = ti.gname = ''
                t.addfile(ti)
            data = files[rel]
            ti = tarfile.TarInfo(rel)
            ti.size = len(data)
            ti.mode = 0o644
            ti.mtime = mtime
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ''
            t.addfile(ti, io.BytesIO(data))
    with open(out_path, 'wb') as f:
        with gzip.GzipFile(fileobj=f, mode='wb', compresslevel=6, mtime=0) as gz:
            gz.write(buf.getvalue())


def pack_run(run_dir: str, out_path: str) -> None:
    run_name = os.path.basename(run_dir.rstrip('/' + BS))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w', format=tarfile.GNU_FORMAT) as t:
        ti = tarfile.TarInfo(run_name + '/')
        ti.type = tarfile.DIRTYPE
        ti.mode = 0o755
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ''
        t.addfile(ti)
        for n in sorted(os.listdir(run_dir)):
            p = os.path.join(run_dir, n)
            if not os.path.isfile(p):
                continue
            with open(p, 'rb') as f:
                data = f.read()
            ti = tarfile.TarInfo(run_name + '/' + n)
            ti.size = len(data)
            ti.mode = 0o644
            ti.mtime = int(os.path.getmtime(p))
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ''
            t.addfile(ti, io.BytesIO(data))
    with open(out_path, 'wb') as f:
        with gzip.GzipFile(fileobj=f, mode='wb', compresslevel=6, mtime=0) as gz:
            gz.write(buf.getvalue())


def write_md5sums(run_dir: str, names: list) -> str:
    lines = []
    for n in names:
        p = os.path.join(run_dir, n)
        if not os.path.isfile(p):
            raise SystemExit('[error] 缺少待登记文件: ' + n)
        lines.append(md5_file(p) + '  ' + n)
    text = '\n'.join(lines) + '\n'
    with open(os.path.join(run_dir, MD5SUMS), 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    return text


# ------------------------------------------------------------------ 主流程
def normalize(run_dir: str, code_source, score, check: bool) -> int:
    log = []

    def say(s=''):
        print(s)
        log.append(str(s))

    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise SystemExit('[error] 目录不存在: ' + run_dir)
    run_name = os.path.basename(run_dir.rstrip('/' + BS))
    outer = run_dir + '.tar.gz'
    manifest = load_manifest(run_dir)

    say('=' * 72)
    say('# 归档规范化: ' + run_name)
    say('=' * 72)
    say('  manifest 条数 : %d' % len(manifest))
    say('  模式          : %s' % ('体检（不写盘）' if check else '写盘'))
    say()

    # ---- 1. code.tar.gz
    say('-- 1. code.tar.gz --')
    code_path = os.path.join(run_dir, CODE_TAR)
    files = {}
    if code_source:
        src = collect_from_source(code_source, manifest)
        missing = [k for k in sorted(manifest) if k not in src]
        mismatch = [k for k in sorted(manifest)
                    if k in src and md5_bytes(src[k]) != manifest[k]]
        say('  来源          : %s' % code_source)
        say('  命中 / 缺失 / 不一致 : %d / %d / %d'
            % (len(manifest) - len(missing) - len(mismatch), len(missing), len(mismatch)))
        if missing:
            say('  !! 缺失: %s' % missing[:20])
        if mismatch:
            say('  !! 内容不一致: %s' % mismatch[:20])
        if missing or mismatch:
            say('  !! 无法忠实重建 code.tar.gz，终止。')
            return 1
        files = {k: src[k] for k in manifest}
        if check:
            say('  [check] 将重建 code.tar.gz（%d 个文件，%d 字节）'
                % (len(files), sum(len(v) for v in files.values())))
        else:
            build_code_tar(files, code_path, run_mtime(run_dir))
            say('  已重建        : %s (%d 字节, md5=%s)'
                % (CODE_TAR, os.path.getsize(code_path), md5_file(code_path)[:12]))
    else:
        say('  未指定 --code-source，跳过重建。')

    if os.path.isfile(code_path):
        got = {}
        with tarfile.open(code_path) as t:
            for m in t.getmembers():
                if m.isfile():
                    got[m.name] = md5_bytes(t.extractfile(m).read())
        ok = [k for k in manifest if got.get(k) == manifest[k]]
        bad = [k for k in manifest if k in got and got[k] != manifest[k]]
        absent = [k for k in manifest if k not in got]
        extra = sorted(set(got) - set(manifest))
        say('  校验 vs manifest: 一致=%d 不一致=%d 缺=%d 多=%d'
            % (len(ok), len(bad), len(absent), len(extra)))
        if bad or absent:
            say('  !! code.tar.gz 与 manifest 不符: bad=%s absent=%s' % (bad[:8], absent[:8]))
        if extra:
            say('  (多出的文件: %s)' % extra[:8])
    else:
        say('  !! code.tar.gz 不存在')
    say()

    # ---- 2. 提交 zip 命名
    say('-- 2. 提交 zip 命名 --')
    sub_zip = find_submission_zip(run_dir)
    target = 'submission.zip'
    if score is not None:
        target = 'submission(%s).zip' % score
    if sub_zip is None:
        say('  !! 目录内没有 submission*.zip')
    else:
        say('  当前          : %s' % sub_zip)
        if sub_zip == target:
            say('  已符合标准    : 无需改名')
        elif check:
            say('  [check] 将改名为 : %s' % target)
        else:
            os.replace(os.path.join(run_dir, sub_zip), os.path.join(run_dir, target))
            say('  已改名        : %s -> %s' % (sub_zip, target))
            sub_zip = target
    say()

    # ---- 3. MD5SUMS
    say('-- 3. MD5SUMS.txt --')
    names = ['predictions.json', 'submission.json', sub_zip,
             'config_snapshot.yaml', 'env_snapshot.json', SNAPSHOT, CODE_TAR]
    names = [n for n in names if n]
    present = [n for n in names if os.path.isfile(os.path.join(run_dir, n))]
    absent = [n for n in names if n not in present]
    for n in present:
        say('  %s  %s' % (md5_file(os.path.join(run_dir, n))[:12], n))
    for n in absent:
        say('  ------------  %s   (缺失，未登记)' % n)
    all_files = sorted(n for n in os.listdir(run_dir)
                       if os.path.isfile(os.path.join(run_dir, n)))
    extra = [n for n in all_files if n not in present and n != MD5SUMS]
    if extra:
        say('  未登记（标准做法：run.log 不入清单）: %s' % extra)
    if check:
        say('  [check] 将按上述顺序重写 ' + MD5SUMS)
    else:
        write_md5sums(run_dir, present)
        say('  已重写        : %s' % MD5SUMS)
    say()

    # ---- 4. 外层回传包
    say('-- 4. 外层回传包 --')
    say('  目标          : %s' % os.path.basename(outer))
    if os.path.isfile(outer):
        with tarfile.open(outer) as t:
            old = sorted(m.name for m in t.getmembers() if m.isfile())
        say('  现有 %d 个文件' % len(old))
        stale = [n for n in old if os.path.basename(n) not in all_files + [MD5SUMS]]
        if stale:
            say('  !! 包内已不在目录中的旧文件: %s' % stale)
    if check:
        say('  [check] 将重新打包')
    else:
        pack_run(run_dir, outer)
        say('  已重新打包    : %s (%d 字节)' % (os.path.basename(outer),
                                              os.path.getsize(outer)))
    say()

    # ---- 5. 复核
    say('-- 5. 复核 --')
    if not check:
        with open(os.path.join(run_dir, MD5SUMS), encoding='utf-8') as f:
            rec = [ln.split() for ln in f.read().splitlines() if ln.strip()]
        bad = []
        for h, n in rec:
            p = os.path.join(run_dir, n)
            if not os.path.isfile(p) or md5_file(p) != h:
                bad.append(n)
        say('  MD5SUMS 逐条核对: %d 条, 失败 %d %s'
            % (len(rec), len(bad), bad or ''))
        with tarfile.open(outer) as t:
            ms = sorted(m.name for m in t.getmembers() if m.isfile())
        want = sorted(run_name + '/' + n for n in os.listdir(run_dir)
                      if os.path.isfile(os.path.join(run_dir, n)))
        say('  回传包文件集与目录一致: %s' % (ms == want))
        if ms != want:
            say('    包内: %s' % ms)
            say('    目录: %s' % want)
    say()
    say('=' * 72)
    say('# 完成' if not check else '# 体检完成（未写盘）')
    say('=' * 72)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description='归档规范化（以标准归档为模板）')
    ap.add_argument('--run', required=True, help='run 归档目录')
    ap.add_argument('--code-source', default=None,
                    help='源码来源（tar.gz 或目录），用于重建 code.tar.gz')
    ap.add_argument('--score', default=None, help='官方成绩，用于 submission(<成绩>).zip 命名')
    ap.add_argument('--check', action='store_true', help='只体检，不写盘')
    args = ap.parse_args()
    return normalize(args.run, args.code_source, args.score, args.check)


if __name__ == '__main__':
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""只读盘点 px-cloud2：模型位置、缓存、conda 环境版本。"""
import os
import paramiko


def main() -> int:
    host = os.environ.get("SSH_HOST", "px-cloud2.matpool.com")
    port = int(os.environ.get("SSH_PORT", "29011"))
    user = os.environ.get("SSH_USER", "root")
    pw = os.environ.get("SSH_PW")
    if not pw:
        print("缺少 SSH_PW")
        return 2
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(host, port=port, username=user, password=pw,
                timeout=25, banner_timeout=30, auth_timeout=30)

    def run(cmd):
        _, out, err = cli.exec_command(cmd, timeout=90)
        return out.read().decode("utf-8", "replace"), \
            err.read().decode("utf-8", "replace")

    print("=== transformers 版本(myconda) ===")
    so, _ = run("/root/miniconda3/envs/myconda/bin/python -c "
                "'import transformers,torch; print(transformers.__version__, torch.__version__)'")
    print(so.strip() or "(无法导入)")
    print("=== 关键包 ===")
    so, _ = run("/root/miniconda3/envs/myconda/bin/pip list 2>/dev/null "
                "| grep -iE 'transformers|^torch|accelerate|peft|safetensors|pillow|opencv' | head -20")
    print(so.strip() or "(n/a)")
    print("=== 全盘找 LocateAnything/Qwen 权重 ===")
    so, _ = run("find / -maxdepth 4 \\( -iname '*locateanything*' -o -iname '*qwen2*' -o -iname '*LocateAnything*' \\) 2>/dev/null | head -20")
    print(so.strip() or "(未找到)")
    print("=== 缓存目录 ===")
    for d in ["/root/.cache/modelscope", "/root/.cache/huggingface",
              "/root/.cache/huggingface/hub", "/models"]:
        so, _ = run(f"ls -d {d} 2>/dev/null && du -sh {d} 2>/dev/null")
        print(so.strip() or f"(无 {d})")
    print("=== /root/.cache/huggingface/hub 内容 ===")
    so, _ = run("ls /root/.cache/huggingface/hub 2>/dev/null | head -20")
    print(so.strip() or "(空)")
    cli.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
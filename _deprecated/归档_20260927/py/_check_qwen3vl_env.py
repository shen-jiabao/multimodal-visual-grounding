# -*- coding: utf-8 -*-
"""检查 venv 依赖是否满足 Qwen3-VL 推理需求。"""
import os
import paramiko

CMD = r'''
P=/srv/venv-locate/bin/python
echo ==VENV_PKGS==
$P -c "import transformers,torch;print('transformers',transformers.__version__);print('torch',torch.__version__,torch.cuda.is_available())" 2>&1 | tail -3
echo ==DETECT==
$P -c "
import transformers
names=[n for n in dir(transformers) if 'qwen' in n.lower()]
print('qwen-related:', sorted(set(names))[:20])
try:
    from transformers import Qwen3VLForConditionalGeneration
    print('Qwen3VL class: OK')
except Exception as e:
    print('Qwen3VL class: NOT AVAILABLE ->', type(e).__name__, e)
" 2>&1 | tail -5
echo ==PIP==
$P -m pip list 2>/dev/null | grep -iE "transformers|accelerate|qwen|torch$|torch |flash|decord|pillow|numpy" | head -15
echo ==MODELSCOPE==
$P -c "import modelscope;print('modelscope',modelscope.__version__)" 2>&1 | tail -1
echo ==MODELDIR==
ls /srv/models/
'''


def main() -> int:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(os.environ["SSH_HOST"], port=int(os.environ["SSH_PORT"]),
                username=os.environ["SSH_USER"],
                password=os.environ["SSH_PW"], timeout=25,
                banner_timeout=30, auth_timeout=30)
    _, out, err = cli.exec_command(CMD, timeout=180)
    print(out.read().decode("utf-8", "replace"))
    e = err.read().decode("utf-8", "replace")
    if e.strip():
        print("--- stderr ---")
        print(e)
    cli.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
import os
import paramiko


def run(cmd):
    _, out, err = cli.exec_command(cmd, timeout=60)
    return out.read().decode("utf-8", "replace"), \
        err.read().decode("utf-8", "replace")


d = r"d:\Desktop\2026AiC"
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["SSH_HOST"], port=int(os.environ["SSH_PORT"]),
            username=os.environ["SSH_USER"],
            password=os.environ["SSH_PW"], timeout=25,
            banner_timeout=30, auth_timeout=30)

so, _ = run('''
echo ==VENV_VER==
/srv/venv-locate/bin/python -c "import transformers,PIL,torch; print(transformers.__version__, PIL.__version__, torch.__version__, torch.cuda.is_available())" 2>&1
echo ==PIP==
/srv/venv-locate/bin/pip list 2>/dev/null | grep -iE "transformers|tokenizers|accelerate|peft|decord|lmdb"
echo ==MODEL==
du -sh /srv/models/LocateAnything-3B 2>/dev/null
ls /srv/models/LocateAnything-3B 2>/dev/null | head -20
echo ==PROC==
pgrep -af snapshot 2>/dev/null | head -3
pgrep -af modelscope 2>/dev/null | head -3
echo ==DISK==
df -h /srv | tail -1
''')
print(so)
cli.close()
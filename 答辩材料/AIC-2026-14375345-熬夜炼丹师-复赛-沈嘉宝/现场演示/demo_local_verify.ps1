# =============================================================================
# 现场演示脚本 · 本地离线核验（答辩用 · Windows PowerShell）
# 用法：在本包根目录执行
#       powershell -ExecutionPolicy Bypass -File 现场演示\demo_local_verify.ps1
# 特点：纯 CPU、无需网络、无需 GPU；①②③ 完全不改动任何文件
# =============================================================================
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$code = Join-Path $root 'AIC-2026-14375345-代码与数据'
Set-Location $code

function Line { Write-Host ('-' * 62) }
function Step($n, $t) { Write-Host ''; Line; Write-Host "【$n】$t" -ForegroundColor Cyan; Line }

Write-Host '======================================================================'
Write-Host ' 答辩现场演示 · 本地离线核验（纯 CPU，无网络，无 GPU）'
Write-Host " $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host '======================================================================'

Step '①' '包级一键自检：指纹 14 项 + 硬约束 4 项 + 改动面读数'
Write-Host '  $ python 自检/_verify_package.py'
python 自检\_verify_package.py
Write-Host ("  退出码 = {0}   （预期 0）" -f $LASTEXITCODE) -ForegroundColor Yellow

Step '②' '结果件四项硬约束（条数 / 键集 / 四字段回显 / bbox 合法性）'
Write-Host '  $ python code/_validate_semi.py ../AIC-2026-14375345-赛题结果文档/result_semi_q38_nonord.zip'
python code\_validate_semi.py "..\AIC-2026-14375345-赛题结果文档\result_semi_q38_nonord.zip"
Write-Host ("  退出码 = {0}   （预期 0）" -f $LASTEXITCODE) -ForegroundColor Yellow

Step '③' '装配器夹具回归：8 组夹具 / 20 项硬断言'
Write-Host '  $ python code/_test_assembly.py'
python code\_test_assembly.py
Write-Host ("  退出码 = {0}   （预期 0）" -f $LASTEXITCODE) -ForegroundColor Yellow

Step '④' '交付件指纹（与 赛题结果文档/SHA256.txt 及 自查报告 一致）'
Write-Host '  $ certutil -hashfile <结果 zip> SHA256'
$zip = Join-Path $root 'AIC-2026-14375345-赛题结果文档\result_semi_q38_nonord.zip'
(Get-FileHash $zip -Algorithm SHA256).Hash
Write-Host '  预期：FB6C2E471A4F5E8907318FF540A5CAABAEC17A29ECAFC17EB651F512A4AE8F35' -ForegroundColor Yellow

Step '⑤' '★ 离线逐条复算交付件（最强一击）—— 需手动执行'
Write-Host @'
  说明：下面这条命令会把交付件从产物**逐条重建**出来。
       产物与提交件 JSON 逐条一致 ⇒ 结果不是手搓的。
  ⚠ 它会在 submission_zips/ 下写出一个重建文件；如需保持交付包原样，
    请先复制整个包到临时目录再执行本条。

  $ python code/_assembly_32b.py --direct _qwen38_direct.json `
        --base result_semi_n32bf16_nonord --base-score 0.7786 `
        --ord-correct 689 --base-correct 3741 `
        --stem result_semi_q38_nonord --zip

  预期输出（六桶互斥，合计 4587 ✓）：
        1 无框 → 保留基线          329
        3 面积>0.35 → 保留基线     100
        5 与基线一致(IoU≥0.5) → 不动  3639
        6 ★改动                    519
        分类合计 4587 = 非序数 4587 ✓（互斥口径自检通过）
        自检：序数层 1103/1103 条与基线逐位相同 ✓
'@ -ForegroundColor Gray

Write-Host ''
Write-Host '======================================================================'
Write-Host ' ①②③④ 全部通过即说明：包未被改动、结果件合规、装配逻辑可复算'
Write-Host '======================================================================'

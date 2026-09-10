param([string]$Python = "python", [int]$Port = 8001)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (!(Test-Path '.env')) {
    Write-Error '请先运行 python scripts/configure.py，再配置数据库和模型。'
    exit 1
}
& $Python -m uvicorn app:app --host 127.0.0.1 --port $Port --ws-max-size 20000
exit $LASTEXITCODE

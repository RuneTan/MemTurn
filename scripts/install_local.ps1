# Run from any directory. Only this checkout's environment/cache/models are written.
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$localPython = Join-Path $projectRoot ".venv-local\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $localPython)) {
    python -m venv (Join-Path $projectRoot ".venv-local")
    if ($LASTEXITCODE -ne 0) { throw "Cannot create local environment" }
}
& $localPython -m pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed" }
& $localPython -m pip install --no-cache-dir -r (Join-Path $projectRoot "configs\local-windows.lock.txt")
if ($LASTEXITCODE -ne 0) { throw "Model dependency installation failed" }
& $localPython (Join-Path $PSScriptRoot "download_local_model.py")
if ($LASTEXITCODE -ne 0) { throw "Pinned model download failed" }

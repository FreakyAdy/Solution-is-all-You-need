# PHANTOM PowerShell Installer for Windows
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  PHANTOM — Universal Model Runtime Platform Installer" -ForegroundColor Cyan
Write-Host "  Run the Unreachable." -ForegroundColor DarkCyan
Write-Host "============================================================"

# Check Python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Error "Python 3.10+ is required on PATH."
    exit 1
}

Write-Host "Installing Python package..." -ForegroundColor Yellow
python -m pip install --upgrade pip
python -m pip install -e python/

# Create PHANTOM home directory
$phantomDir = Join-Path $env:USERPROFILE ".phantom"
New-Item -ItemType Directory -Force -Path (Join-Path $phantomDir "models") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $phantomDir "downloads") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $phantomDir "logs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $phantomDir "rag") | Out-Null

Write-Host "Running system diagnostics..." -ForegroundColor Yellow
python -m phantom.phantom_cli doctor

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "✓ PHANTOM is ready on Windows!" -ForegroundColor Green
Write-Host "Try running:" -ForegroundColor White
Write-Host "    phantom plan llama3:70b" -ForegroundColor White
Write-Host "    phantom serve" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Green

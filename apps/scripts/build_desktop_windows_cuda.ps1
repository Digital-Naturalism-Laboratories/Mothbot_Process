$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
Set-Location $RootDir
$AppsDir = Join-Path $RootDir "apps"
$BuildDir = Join-Path $AppsDir "build"
$DistDir = Join-Path $AppsDir "dist"

$VenvDir = Join-Path $RootDir ".venv-packaging"
python -m venv $VenvDir

$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
New-Item -ItemType Directory -Force -Path $DistDir | Out-Null

$env:ULTRALYTICS_AUTOINSTALL = "0"

& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -e ".[cuda118-packaging]" --extra-index-url https://download.pytorch.org/whl/cu118
& $PythonExe -m PyInstaller --clean --noconfirm --workpath $BuildDir --distpath $DistDir apps/packaging/pyinstaller/mothbot_desktop.spec

& "$RootDir\apps\scripts\package_release_windows_cuda.ps1"

Write-Host ""
Write-Host "Build complete."
Write-Host "Artifact: $DistDir\Mothbot"

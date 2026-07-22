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
if ($env:GITHUB_REF_NAME -and $env:GITHUB_REF_NAME.StartsWith("v")) {
  $env:MOTHBOT_RELEASE_VERSION = $env:GITHUB_REF_NAME.Substring(1)
} else {
  $env:MOTHBOT_RELEASE_VERSION = python -c "import tomllib, pathlib; p=pathlib.Path('pyproject.toml'); print(tomllib.loads(p.read_text())['project']['version'])"
}
$VersionFile = Join-Path $BuildDir "VERSION"
Set-Content -Path $VersionFile -Value $env:MOTHBOT_RELEASE_VERSION -NoNewline
$env:MOTHBOT_VERSION_FILE = $VersionFile

& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -e ".[cuda118-packaging]" --extra-index-url https://download.pytorch.org/whl/cu118

Write-Host "=== Verifying torch installation ==="
& $PythonExe -c "import torch; print('torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Fetch large models (e.g. the default birefnet bg-removal model) into assets/
# so the PyInstaller spec can bundle them into the app.
& $PythonExe apps/scripts/fetch_bundled_models.py

& $PythonExe -m PyInstaller --clean --noconfirm --workpath $BuildDir --distpath $DistDir apps/packaging/pyinstaller/mothbot_desktop.spec

& "$RootDir\apps\scripts\package_release_windows_cuda.ps1"

Write-Host ""
Write-Host "Build complete."
Write-Host "Artifact: $DistDir\Mothbot"

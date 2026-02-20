$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
Set-Location $RootDir
$AppsDir = Join-Path $RootDir "apps"
$DistDir = Join-Path $AppsDir "dist"
$ReleaseDir = Join-Path $AppsDir "release"

$ExecutablePath = Join-Path $DistDir "Mothbot\Mothbot.exe"
if (-not (Test-Path $ExecutablePath)) {
  throw "Missing Windows executable at $ExecutablePath. Run: .\apps\scripts\build_desktop_windows.ps1"
}

$SevenZip = Get-Command 7z -ErrorAction SilentlyContinue
if ($null -eq $SevenZip) {
  throw "7z is required to package Windows release artifacts."
}

$Version = python -c "import tomllib, pathlib; p=pathlib.Path('pyproject.toml'); print(tomllib.loads(p.read_text())['project']['version'])"
$Arch = $env:PROCESSOR_ARCHITECTURE.ToLower()
$TargetPath = Join-Path $ReleaseDir "Mothbot-$Version-windows-$Arch.zip"

New-Item -ItemType Directory -Force -Path $ReleaseDir | Out-Null
if (Test-Path $TargetPath) {
  Remove-Item $TargetPath -Force
}

# Keep .zip output for user compatibility but use 7z compression for better ratios.
& $SevenZip.Source a -tzip -mx=9 $TargetPath (Join-Path $DistDir "Mothbot\*") | Out-Null

$MaxBytes = 1932735283
$WarnBytes = 1717986918
$FileInfo = Get-Item $TargetPath
$FileSizeBytes = [int64]$FileInfo.Length
$FileSizeHuman = "{0:N2} GiB" -f ($FileSizeBytes / 1GB)

if ($env:GITHUB_STEP_SUMMARY) {
  @"
### Windows artifact size

| Artifact | Size |
| --- | --- |
| $($FileInfo.Name) | $FileSizeHuman |
"@ | Out-File -FilePath $env:GITHUB_STEP_SUMMARY -Encoding utf8 -Append
}

Write-Host "Windows release artifact size: $FileSizeHuman"
if ($FileSizeBytes -ge $WarnBytes) {
  Write-Host "::warning::Windows release artifact is above 1.6 GiB ($FileSizeHuman)."
}
if ($FileSizeBytes -ge $MaxBytes) {
  throw "Windows release artifact exceeds 1.8 GiB ($FileSizeHuman)."
}

Write-Host ""
Write-Host "Release artifact created:"
Get-Item $TargetPath | Select-Object Name, Length

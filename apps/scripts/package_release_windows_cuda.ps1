$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
Set-Location $RootDir
$AppsDir = Join-Path $RootDir "apps"
$DistDir = Join-Path $AppsDir "dist"
$ReleaseDir = Join-Path $AppsDir "release"
$ExecutablePath = Join-Path $DistDir "Mothbot\Mothbot.exe"
if (-not (Test-Path $ExecutablePath)) {
  throw "Missing Windows executable at $ExecutablePath. Run: .\apps\scripts\build_desktop_windows_cuda.ps1"
}
$SevenZip = Get-Command 7z -ErrorAction SilentlyContinue
if ($null -eq $SevenZip) {
  throw "7z is required to package Windows release artifacts."
}
$Version = $env:MOTHBOT_RELEASE_VERSION
if ([string]::IsNullOrWhiteSpace($Version)) {
  $Version = python -c "import tomllib, pathlib; p=pathlib.Path('pyproject.toml'); print(tomllib.loads(p.read_text())['project']['version'])"
}
$Arch = $env:PROCESSOR_ARCHITECTURE.ToLower()
$TargetPath = Join-Path $ReleaseDir "Mothbot-$Version-windows-$Arch-cuda118.zip"
New-Item -ItemType Directory -Force -Path $ReleaseDir | Out-Null
# Clean up any previous parts
Get-ChildItem "$TargetPath*" -ErrorAction SilentlyContinue | Remove-Item -Force
# Create split zip archive
& $SevenZip.Source a -tzip -mx=5 -v800m $TargetPath (Join-Path $DistDir "Mothbot\*") | Out-Null
# Debug: show what was created
Write-Host "Files in release dir:"
Get-ChildItem $ReleaseDir | Select-Object Name, Length
if ($env:GITHUB_STEP_SUMMARY) {
    $parts = Get-ChildItem "$TargetPath.*" | Sort-Object Name
    $totalBytes = ($parts | Measure-Object -Property Length -Sum).Sum
    $totalHuman = "{0:N2} GiB" -f ($totalBytes / 1GB)
    @"
### Windows CUDA artifact size
| Part | Size |
| --- | --- |
$(($parts | ForEach-Object { "| $($_.Name) | $("{0:N0} MB" -f ($_.Length / 1MB)) |" }) -join "`n")
| **Total** | **$totalHuman** |
"@ | Out-File -FilePath $env:GITHUB_STEP_SUMMARY -Encoding utf8 -Append
}
Write-Host ""
Write-Host "Release artifact parts created:"
Get-ChildItem "$TargetPath.*" | Sort-Object Name | Select-Object Name, Length
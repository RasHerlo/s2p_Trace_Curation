# Build the Windows desktop app: portable zip + (if Inno Setup is installed) an installer.
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
#
# Locally this uses a separate build venv (venv_build\) so the dev venv is never touched.
# In CI (or with -UseCurrentPython) it uses the python on PATH as-is.

param(
    [switch]$UseCurrentPython
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$AppExeName = "s2pTraceCuration"

if ($UseCurrentPython -or $env:CI) {
    $PythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCmd) { throw "Python is not on PATH." }
    $Python = $PythonCmd.Source
} else {
    $BuildVenv = Join-Path $Root "venv_build"
    $Python = Join-Path $BuildVenv "Scripts\python.exe"
    if (-not (Test-Path $Python)) {
        $BasePython = if ($env:S2P_BUILD_PYTHON) { $env:S2P_BUILD_PYTHON } else { (Get-Command python).Source }
        Write-Host "Creating build venv from $BasePython"
        & $BasePython -m venv $BuildVenv
        if ($LASTEXITCODE -ne 0) { throw "Could not create build venv." }
    }
    & $Python -m pip install --upgrade pip
    & $Python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Installing requirements failed." }
}

Write-Host "Using Python: $Python"
& $Python -c "import struct, sys; assert struct.calcsize('P') == 8, '64-bit Python required'; print(sys.version)"
if ($LASTEXITCODE -ne 0) { throw "A 64-bit Python is required." }
$Version = (& $Python -c "from s2p_trace_curation import __version__; print(__version__)").Trim()
Write-Host "Building s2p Trace Curation $Version"

Write-Host "Installing/updating PyInstaller..."
& $Python -m pip install -r requirements-build.txt
if ($LASTEXITCODE -ne 0) { throw "Installing build requirements failed." }

Write-Host "Running PyInstaller..."
& $Python -m PyInstaller --noconfirm --clean --distpath dist --workpath build "packaging\$AppExeName.spec"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }

$AppDir = Join-Path $Root "dist\$AppExeName"
$AppExe = Join-Path $AppDir "$AppExeName.exe"
if (-not (Test-Path $AppExe)) { throw "Expected $AppExeName.exe under $AppDir" }

Write-Host "Running selftest on the built app..."
$Proc = Start-Process -FilePath $AppExe -ArgumentList "--selftest" -Wait -PassThru
if ($Proc.ExitCode -ne 0) {
    throw "Selftest failed (exit $($Proc.ExitCode)). See $HOME\.s2p_trace_curation\error.log"
}
Write-Host "Selftest passed."

$Readme = Join-Path $AppDir "README.txt"
@"
s2p Trace Curation $Version (portable)

Double-click $AppExeName.exe
No Python install is required.

Use File > Open suite2p folder... and select a suite2p directory
(the folder that contains plane0/).

Settings and error logs are kept in %USERPROFILE%\.s2p_trace_curation\
"@ | Set-Content -Path $Readme -Encoding UTF8

$ReleaseDir = Join-Path $Root "dist\release"
New-Item -ItemType Directory -Force -Path $ReleaseDir | Out-Null

$ZipPath = Join-Path $ReleaseDir "$AppExeName-$Version-windows-portable.zip"
if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }

Write-Host "Creating portable zip..."
Compress-Archive -Path $AppDir -DestinationPath $ZipPath -Force

$IsccCandidates = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles}\Inno Setup 6\ISCC.exe",
    "${env:LOCALAPPDATA}\Programs\Inno Setup 6\ISCC.exe"
)
$Iscc = $IsccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($Iscc) {
    Write-Host "Building installer with $Iscc"
    & $Iscc "/DMyAppVersion=$Version" (Join-Path $Root "packaging\installer.iss")
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed." }
} else {
    Write-Host "Inno Setup 6 not found. Portable zip was created; skip the installer or install Inno Setup and re-run."
}

Write-Host ""
Write-Host "Release files in dist\release :"
Get-ChildItem $ReleaseDir | ForEach-Object { Write-Host " - $($_.Name)" }

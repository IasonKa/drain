# One-time set-up of the Kratos drain CFD on a fresh Windows 10/11 x64 machine.
# Run it through SETUP.bat (double-click), or:
#   powershell -ExecutionPolicy Bypass -File vof\kratos\setup_windows.ps1
#
# Installs, all from the official sources:
#   1. Microsoft Visual C++ 2015-2022 runtime (x64): Kratos needs MSVCP140/VCOMP140. This is the
#      only step that asks for administrator approval (a Windows prompt), and only if it is missing.
#   2. Python 3.12 for the current user (python.org installer, no admin rights), if missing.
#   3. A Python environment in %USERPROFILE%\venvs\drain-kratos with Kratos Multiphysics, gmsh,
#      numpy, matplotlib, pyvista (requirements-windows.txt), about 0.7 GB.
# Then it checks that Kratos and gmsh load.
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $env:USERPROFILE "venvs\drain-kratos"
$Tmp = Join-Path $env:TEMP "drain-kratos-setup"
New-Item -ItemType Directory -Force $Tmp | Out-Null
function Step($t) { Write-Host ""; Write-Host "== $t" -ForegroundColor Cyan }

# 1. Visual C++ runtime ------------------------------------------------------------------
Step "Microsoft Visual C++ runtime"
$sys = Join-Path $env:WINDIR "System32"
if ((Test-Path "$sys\msvcp140.dll") -and (Test-Path "$sys\vcomp140.dll")) {
    Write-Host "already installed"
} else {
    $vc = Join-Path $Tmp "vc_redist.x64.exe"
    Write-Host "downloading https://aka.ms/vs/17/release/vc_redist.x64.exe"
    Invoke-WebRequest "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile $vc -UseBasicParsing
    Write-Host "installing (Windows will ask for administrator approval)"
    $p = Start-Process $vc -ArgumentList "/install", "/quiet", "/norestart" -Verb RunAs -Wait -PassThru
    if ($p.ExitCode -ne 0 -and $p.ExitCode -ne 3010 -and $p.ExitCode -ne 1638) {
        throw "Visual C++ runtime installer failed with exit code $($p.ExitCode)"
    }
}

# 2. Python 3.12 ------------------------------------------------------------------------
Step "Python 3.12"
$Py = $null
$cand = @((Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
          "C:\Program Files\Python312\python.exe")
foreach ($c in $cand) { if (Test-Path $c) { $Py = $c; break } }
if (-not $Py) {
    try { $Py = (& py -3.12 -c "import sys; print(sys.executable)" 2>$null) } catch { $Py = $null }
}
if (-not $Py -or -not (Test-Path $Py)) {
    $inst = Join-Path $Tmp "python-3.12.10-amd64.exe"
    Write-Host "downloading Python 3.12.10 from python.org"
    Invoke-WebRequest "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe" -OutFile $inst -UseBasicParsing
    Write-Host "installing for the current user (no admin rights needed)"
    $p = Start-Process $inst -ArgumentList "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_launcher=1",
                                           "InstallLauncherAllUsers=0", "Include_test=0" -Wait -PassThru
    if ($p.ExitCode -ne 0) { throw "Python installer failed with exit code $($p.ExitCode)" }
    $Py = $cand[0]
}
Write-Host "using $Py"

# 3. Python environment with Kratos ------------------------------------------------------
Step "Python environment $Venv"
if (-not (Test-Path "$Venv\Scripts\python.exe")) { & $Py -m venv $Venv }
$VPy = "$Venv\Scripts\python.exe"
& $VPy -m pip install --upgrade pip --quiet
& $VPy -m pip install -r (Join-Path $Here "requirements-windows.txt")
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# 4. Check --------------------------------------------------------------------------------
Step "Check"
& $VPy -c "import KratosMultiphysics as KM, KratosMultiphysics.FluidDynamicsApplication, gmsh, pyvista; print('Kratos OK, threads', KM.ParallelUtilities.GetNumThreads(), '- gmsh', gmsh.__version__, '- pyvista', pyvista.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Kratos does not load - see the messages above" }
Write-Host ""
Write-Host "Set-up finished. Now double-click RUN.bat (or run vof\kratos\run_all.ps1)." -ForegroundColor Green

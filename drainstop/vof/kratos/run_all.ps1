# Build, mesh, run and picture the Kratos drain CFD cases. Run SETUP.bat once first.
#   RUN.bat                                          (double-click: both cases, all CPU cores)
#   powershell -ExecutionPolicy Bypass -File vof\kratos\run_all.ps1 -Cases rec_tilt10:8
#   ... -Threads 16 -Force -FineUSZ
#
# Cases are <template>:<pool level in mm>; templates: rec_tilt10 (recommended layout) and usz_tilt10
# (as-built USZ field), both with the outlet end 10 deg up. A finished case (FINISHED in its run folder)
# is not run again unless -Force. Pictures go to vof\kratos\results\<case>\.
param(
    [string[]]$Cases = @("rec_tilt10:8", "usz_tilt10:21"),
    [int]$Threads = 0,          # 0: all physical cores
    [switch]$Force,
    [switch]$FineUSZ            # USZ with 0.25 mm opening cells (about 1.7x the cost) instead of 0.35 mm
)
$ErrorActionPreference = "Continue"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = (Resolve-Path (Join-Path $Here "..\..")).Path            # drainstop\
$VPy = Join-Path $env:USERPROFILE "venvs\drain-kratos\Scripts\python.exe"
if (-not (Test-Path $VPy)) { Write-Host "Run SETUP.bat first." -ForegroundColor Red; exit 1 }
if ($Threads -le 0) { $Threads = (Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum }
$env:PYTHONUTF8 = "1"
Set-Location $Root

# mesh and run settings per template (see README.md)
$Presets = @{
    "rec_tilt10" = @("--hole", "0.25", "--end", "0.10")
    "usz_tilt10" = $(if ($FineUSZ) { @("--hole", "0.25", "--end", "0.06") }
                     else { @("--hole", "0.35", "--skin", "0.6", "--curv", "16", "--end", "0.06") })
}

function Run-Py([string]$log, [string[]]$argv) {
    # run python with these arguments; show and log its output; stop on failure
    & $VPy -u @argv 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { throw "failed: python $($argv -join ' ') (see $log)" }
}

foreach ($c in $Cases) {
    $tmpl, $h = $c.Split(":")
    $name = "${tmpl}_h$h" + $(if ($tmpl -eq "usz_tilt10" -and -not $FineUSZ) { "_coarse" } else { "" })
    $run = Join-Path $Root "vof\kratos\runs\$name"
    $log = Join-Path $Root "vof\kratos\runs\$name.log"
    New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
    Write-Host ""
    Write-Host "== $name   ($Threads threads, $(Get-Date -Format 'yyyy-MM-dd HH:mm'))" -ForegroundColor Cyan
    try {
        if ((Test-Path "$run\FINISHED") -and -not $Force) {
            Write-Host "already finished; use -Force to run it again"
        } else {
            if ($Force -or -not (Test-Path "$run\drain.mdpa")) {
                Write-Host "-- building the geometry and mesh (1 min for rec_tilt10, ~15 min for usz_tilt10)"
                Run-Py $log (@("vof\kratos\make_case.py", $tmpl, $h) + $Presets[$tmpl] + @("--name", $name))
            }
            # a run stopped part-way (reboot, closed window) continues from its last 5 ms snapshot
            $resume = $(if ($Force) { @() } else { @("--resume") })
            Write-Host "-- running Kratos (progress every 10 steps; this takes hours)"
            Run-Py $log (@("vof\kratos\run_case.py", $run, "--threads", "$Threads") + $resume)
        }
        Write-Host "-- pictures"
        Run-Py $log @("vof\kratos\show_setup.py", $run)
        Run-Py $log @("vof\kratos\render_kratos.py", $run, "--frames")
        Run-Py $log @("vof\kratos\make_gifs.py", $run)
        $res = Join-Path $Root "vof\kratos\results\$name"
        New-Item -ItemType Directory -Force $res | Out-Null
        Copy-Item "$run\pictures\*.png", "$run\pictures\*.gif", "$run\pictures\summary.json", "$run\model_state.json",
                  "$run\mesh_info.json", "$run\monitor.csv" -Destination $res -ErrorAction SilentlyContinue
        Write-Host "results: $res" -ForegroundColor Green
    } catch {
        Write-Host $_ -ForegroundColor Red
    }
}
Write-Host ""
Write-Host "Done. Pictures are in vof\kratos\results\. To share them: commit and push that folder." -ForegroundColor Green

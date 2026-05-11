<#
.SYNOPSIS
    Launch N Trackmania + TMInterface instances, each on a unique port,
    then start Python training.

.DESCRIPTION
    Because all TM windows share the same Plugins folder, every instance
    would normally load the same Python_Link.as and try to bind port 8483.

    This script works by:
      1. Reading the current Python_Link.as as a template.
      2. For each desired port, writing a version with that port hardcoded
         into the RegisterVariable("custom_port", PORT) default.
      3. Copying it to the TMInterface Plugins folder.
      4. Launching TM and waiting for it to fully load (plugin binds port).
      5. Repeating for the next port.
      6. Restoring the original Python_Link.as.
      7. Optionally starting training via main.py.

    The AUTO-SCAN feature in Python_Link.as is a cleaner alternative that
    works without this script -- see MULTI_INSTANCE_GUIDE.md Method 1.

.PARAMETER Ports
    List of ports, one per TM instance.  Default: 8483 8484

.PARAMETER TmExe
    Full path to TmForever.exe.  Detected automatically from common locations.

.PARAMETER WaitSeconds
    Seconds to wait after launching each TM window for it to fully load
    and the plugin to bind the port.  Default: 12

.PARAMETER StartTraining
    If set, runs  python main.py --ports <Ports>  after all instances start.

.EXAMPLE
    .\launch_multienv.ps1
    .\launch_multienv.ps1 -Ports 8483,8484,8485 -StartTraining
    .\launch_multienv.ps1 -Ports 8483,8484 -WaitSeconds 15 -StartTraining
#>

param(
    [int[]]$Ports        = @(8483, 8484),
    [string]$TmExe       = "",
    [int]$WaitSeconds    = 12,
    [switch]$StartTraining
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Locate TmForever.exe
# ---------------------------------------------------------------------------
if (-not $TmExe) {
    $candidates = @(
        "$env:USERPROFILE\AppData\Local\TrackMania\TmForever.exe",
        "C:\Program Files\TrackMania Nations Forever\TmForever.exe",
        "C:\Program Files (x86)\TrackMania Nations Forever\TmForever.exe",
        "C:\Games\TrackMania Nations Forever\TmForever.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { $TmExe = $c; break }
    }
}
if (-not $TmExe -or -not (Test-Path $TmExe)) {
    Write-Error @"
Could not locate TmForever.exe.  Pass the full path explicitly:
    .\launch_multienv.ps1 -TmExe "C:\path\to\TmForever.exe"
"@
}

# ---------------------------------------------------------------------------
# Locate Plugins folder (TMInterface places Python_Link.as here)
# ---------------------------------------------------------------------------
$pluginsDst = "$env:USERPROFILE\Documents\TMInterface\Plugins"
if (-not (Test-Path $pluginsDst)) {
    New-Item -ItemType Directory -Path $pluginsDst -Force | Out-Null
    Write-Host "[INFO] Created Plugins folder at $pluginsDst"
}
$pluginTarget = Join-Path $pluginsDst "Python_Link.as"

# ---------------------------------------------------------------------------
# Source Python_Link.as template (repo copy)
# ---------------------------------------------------------------------------
$scriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot   = Split-Path -Parent $scriptDir
$pluginSrc  = Join-Path $repoRoot "interfacing\Python_Link.as"

if (-not (Test-Path $pluginSrc)) {
    Write-Error "Could not find $pluginSrc – run from the repo root or check the path."
}

$originalContent = Get-Content $pluginSrc -Raw

# ---------------------------------------------------------------------------
# Launch one TM instance per port
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=========================================="
Write-Host "  Launching $($Ports.Count) TM instance(s)"
Write-Host "  Ports: $($Ports -join ', ')"
Write-Host "  TmExe: $TmExe"
Write-Host "=========================================="
Write-Host ""

foreach ($port in $Ports) {
    Write-Host "[Port $port] Patching Python_Link.as ..."

    # Replace the default value in RegisterVariable("custom_port", DEFAULT)
    # The pattern matches any existing default integer.
    $patched = $originalContent -replace `
        'RegisterVariable\("custom_port",\s*\d+\)', `
        "RegisterVariable(`"custom_port`", $port)"

    # Also patch the auto-scan loop so it starts from this specific port
    # (optional safety measure for the auto-scan fallback).
    $patched = $patched -replace `
        'for \(uint16 p = 8483;', `
        "for (uint16 p = $port;"

    Set-Content -Path $pluginTarget -Value $patched -Encoding UTF8

    Write-Host "[Port $port] Launching TM ..."
    Start-Process -FilePath $TmExe
    Write-Host "[Port $port] Waiting $WaitSeconds s for plugin to load and bind port ..."
    Start-Sleep -Seconds $WaitSeconds
    Write-Host "[Port $port] Ready."
    Write-Host ""
}

# ---------------------------------------------------------------------------
# Restore original plugin file
# ---------------------------------------------------------------------------
Write-Host "Restoring original Python_Link.as ..."
Set-Content -Path $pluginTarget -Value $originalContent -Encoding UTF8
Write-Host "Done."
Write-Host ""

# ---------------------------------------------------------------------------
# Optional: start training
# ---------------------------------------------------------------------------
if ($StartTraining) {
    $portList = $Ports -join " "
    Write-Host "Starting training: python main.py --ports $portList"
    Write-Host ""
    Push-Location $repoRoot
    python main.py --ports $portList
    Pop-Location
} else {
    $portList = $Ports -join " "
    Write-Host "All instances are running.  Now:"
    Write-Host "  1. In each TM window, load your training map."
    Write-Host "  2. Run:  python main.py --ports $portList"
    Write-Host ""
    Write-Host "Or pass -StartTraining to do both automatically (map must already be loaded)."
}

<#
.SYNOPSIS
  Start Docker Desktop (WSL 2 backend) on a Windows machine where it crashes on every restart.

.DESCRIPTION
  Docker Desktop 4.73 on Windows leaves AF_UNIX socket files behind when it stops (even on a
  clean stop) and then cannot delete them on the next start. Windows reports "The file cannot
  be accessed by the system" (error 1920), and Docker Desktop dies with
    starting services: initializing Inference manager: listening on unix://...\dockerInference
  or
    starting services: initializing Secrets Engine: listening on unix://...\engine.sock
  A forced kill can also leave Docker's WSL data disk attached
  (WSL_E_USER_VHD_ALREADY_ATTACHED), so the next start cannot mount it.

  Windows cannot open those socket files, but WSL can unlink them. This script
    1. returns early if the Docker engine already answers,
    2. stops any half-started Docker Desktop (not the Windows service),
    3. terminates only the docker-desktop WSL distro and detaches its data disk,
    4. deletes the stale socket files through WSL,
    5. starts Docker Desktop and waits for the engine.
  It never touches containers, images or volumes, and no WSL distro other than
  docker-desktop is stopped.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\windows\start-docker-desktop.ps1
#>
param(
  [string]$LocalAppData = $env:LOCALAPPDATA,
  [int]$TimeoutSeconds = 240
)

$ErrorActionPreference = "Continue"
$dockerDesktop = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"

function Test-Engine {
  $v = (& docker version --format "{{.Server.Version}}" 2>$null | Out-String).Trim()
  return $v -match '^\d'
}

if (Test-Engine) {
  Write-Host "Docker engine is already running."
  exit 0
}

Write-Host "1. stopping any half-started Docker Desktop (the Windows service is left alone)"
Get-Process | Where-Object {
  $_.Name -like "Docker Desktop*" -or ($_.Name -like "com.docker.*" -and $_.Name -ne "com.docker.service")
} | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

Write-Host "2. terminating the docker-desktop WSL distro and detaching its data disk"
wsl.exe --terminate docker-desktop 2>&1 | Out-Null
$vhd = Join-Path $LocalAppData "Docker\wsl\disk\docker_data.vhdx"
if (Test-Path -LiteralPath $vhd) { wsl.exe --unmount $vhd 2>&1 | Out-Null }

Write-Host "3. deleting stale socket files (through WSL: Windows cannot open them)"
# Windows cannot even list these files (Get-ChildItem skips them), so the whole
# listing + deletion is done inside WSL by a small shell script.
$base = "/mnt/" + $LocalAppData.Substring(0, 1).ToLower() + ($LocalAppData.Substring(2) -replace '\\', '/')
$script = [IO.File]::ReadAllText((Join-Path $PSScriptRoot "clear-docker-sockets.sh")).TrimStart([char]0xFEFF) -replace "`r", ""
$script | wsl.exe -e sh -s -- "$base"

Write-Host "4. starting Docker Desktop"
Start-Process $dockerDesktop
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
  if (Test-Engine) {
    $v = (& docker version --format "{{.Server.Version}}" 2>$null | Out-String).Trim()
    Write-Host "Docker engine $v is up."
    exit 0
  }
  Start-Sleep -Seconds 5
}

Write-Host "Docker did not come up within $TimeoutSeconds s. Last backend errors:"
$log = Join-Path $LocalAppData "Docker\log\host\com.docker.backend.exe.log"
if (Test-Path -LiteralPath $log) {
  Select-String -LiteralPath $log -Pattern "backend cancelling|backend crashed|Failed to attach|exit status" |
    Select-Object -Last 4 | ForEach-Object { "  " + $_.Line.Substring(0, [Math]::Min(300, $_.Line.Length)) }
}
exit 1

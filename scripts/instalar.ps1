# =============================================================================
#  Instalador para Windows - Sistema de prevención de caídas en escaleras
# =============================================================================
#  Lo lanza instalar.bat (doble clic). Se puede repetir: lo que ya está
#  instalado no se vuelve a instalar. Todo queda DENTRO de la carpeta del
#  proyecto (no toca el sistema ni necesita administrador):
#     .tools\uv.exe   gestor de Python y paquetes (uv, de Astral)
#     .python\        Python 3.12
#     .venv\          entorno del proyecto
#     .cache\         caché de descargas
#  Al final crea iniciar.bat y un acceso directo en el Escritorio.
#
#  Opciones:  instalar.bat --cpu        fuerza PyTorch sin GPU
# =============================================================================
param([switch]$cpu)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "================================================================"
Write-Host " Instalador - Sistema de prevencion de caidas en escaleras"
Write-Host "================================================================"
Write-Host "Carpeta: $root"

# Todo dentro del proyecto
$env:UV_CACHE_DIR = Join-Path $root ".cache\uv"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $root ".python"
$env:UV_HTTP_TIMEOUT = "600"
# Copiar en vez de enlazar: los discos exFAT/FAT32 (p. ej. memorias USB o
# discos externos) no admiten enlaces duros ni de carpeta.
$env:UV_LINK_MODE = "copy"
# Ultralytics guarda su configuración aquí y no en %APPDATA%
$env:YOLO_CONFIG_DIR = Join-Path $root ".cache\ultralytics"
$env:PYTHONNOUSERSITE = "1"

# --- uv ----------------------------------------------------------------------
Write-Host "`n== uv (gestor de Python y paquetes)"
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { $uv = Join-Path $root ".tools\uv.exe" }
if (-not (Test-Path $uv)) {
    $arch = if ([Environment]::Is64BitOperatingSystem) {
        if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "aarch64" } else { "x86_64" }
    } else { "i686" }
    $zipUrl = "https://github.com/astral-sh/uv/releases/latest/download/uv-$arch-pc-windows-msvc.zip"
    Write-Host "  descargando $zipUrl"
    $tools = Join-Path $root ".tools"
    New-Item -ItemType Directory -Force $tools | Out-Null
    $zip = Join-Path $tools "uv.zip"
    Invoke-WebRequest $zipUrl -OutFile $zip -UseBasicParsing
    Expand-Archive $zip -DestinationPath $tools -Force
    Remove-Item $zip
    $uv = Join-Path $tools "uv.exe"
}
& $uv --version

# --- Python 3.12 y entorno ---------------------------------------------------
Write-Host "`n== Python 3.12"
& $uv python install 3.12
$pyExit = $LASTEXITCODE
# Python descargado dentro del proyecto (.python\cpython-3.12.x-...)
$pyHome = Get-ChildItem (Join-Path $root ".python") -Directory -Filter "cpython-3.12*" -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName "python.exe") } |
    Sort-Object Name -Descending | Select-Object -First 1
if ($pyExit -ne 0) {
    # En discos exFAT/FAT32 uv descarga Python pero no puede crear el enlace
    # de carpeta "3.12" ("Funcion incorrecta"): se usa la instalacion directa.
    if ($pyHome) {
        Write-Host "  aviso: el disco no admite enlaces (exFAT/FAT32); se usa $($pyHome.Name)"
    } else { throw "No se pudo instalar Python 3.12" }
}
$pySpec = if ($pyHome) { Join-Path $pyHome.FullName "python.exe" } else { "3.12" }
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "  creando entorno .venv"
    & $uv venv .venv --python $pySpec --seed
    if ($LASTEXITCODE -ne 0) { throw "No se pudo crear .venv" }
} else { Write-Host "  entorno .venv ya existe" }

# --- paquetes, modelos, diagnóstico, iniciar.bat -----------------------------
$installArgs = @("tools\install.py", "--uv", $uv)
if ($cpu) { $installArgs += "--cpu" }
& $py @installArgs
if ($LASTEXITCODE -ne 0) { throw "La instalación no terminó (ver mensajes de arriba)" }

# --- acceso directo en el Escritorio -----------------------------------------
Write-Host "`n== Acceso directo"
$desktop = [Environment]::GetFolderPath("Desktop")
$lnk = Join-Path $desktop "Sistema Escaleras.lnk"
$sh = New-Object -ComObject WScript.Shell
$s = $sh.CreateShortcut($lnk)
$s.TargetPath = Join-Path $root "iniciar.bat"
$s.WorkingDirectory = $root
$s.Description = "Arranca el sistema de prevencion de caidas y abre el panel"
$s.IconLocation = "$env:SystemRoot\System32\shell32.dll,137"
$s.Save()
Write-Host "  creado: $lnk"

Write-Host "`n================================================================"
Write-Host " Instalacion completa."
Write-Host " Doble clic en 'Sistema Escaleras' (Escritorio) o en iniciar.bat"
Write-Host "================================================================"

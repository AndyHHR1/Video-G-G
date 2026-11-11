#!/usr/bin/env bash
# =============================================================================
#  Instalador para Linux / WSL / macOS - Sistema de prevención de caídas
# =============================================================================
#    ./instalar.sh          GPU NVIDIA si hay, si no CPU
#    ./instalar.sh --cpu    fuerza PyTorch sin GPU
#
#  Se puede repetir: lo que ya está instalado no se vuelve a instalar. Todo
#  queda DENTRO de la carpeta del proyecto (no toca el sistema, sin sudo):
#     .tools/uv   .python/   .venv/   .cache/
#  Al final crea iniciar.sh y un acceso directo (menú de aplicaciones y
#  Escritorio; en WSL, también en el Escritorio de Windows).
# =============================================================================
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
ROOT="$(pwd)"
CPU=""
[ "${1:-}" = "--cpu" ] && CPU="--cpu"

echo "================================================================"
echo " Instalador - Sistema de prevención de caídas en escaleras"
echo "================================================================"
echo "Carpeta: $ROOT"

export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/.python"
export UV_HTTP_TIMEOUT=600
# copiar en vez de enlazar: funciona también en discos exFAT/FAT32
export UV_LINK_MODE=copy
export YOLO_CONFIG_DIR="$ROOT/.cache/ultralytics"   # config de Ultralytics, dentro del proyecto
export PYTHONNOUSERSITE=1

# --- uv ----------------------------------------------------------------------
echo; echo "== uv (gestor de Python y paquetes)"
UV="$(command -v uv || true)"
[ -n "$UV" ] || UV="$ROOT/.tools/uv"
if [ ! -x "$UV" ]; then
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64)  T=x86_64-unknown-linux-gnu ;;
        Linux-aarch64) T=aarch64-unknown-linux-gnu ;;
        Darwin-arm64)  T=aarch64-apple-darwin ;;
        Darwin-x86_64) T=x86_64-apple-darwin ;;
        *) echo "Sistema no soportado: $(uname -sm)"; exit 1 ;;
    esac
    URL="https://github.com/astral-sh/uv/releases/latest/download/uv-$T.tar.gz"
    echo "  descargando $URL"
    mkdir -p "$ROOT/.tools"
    if command -v curl >/dev/null; then curl -LsSf "$URL" -o "$ROOT/.tools/uv.tgz"
    else wget -qO "$ROOT/.tools/uv.tgz" "$URL"; fi
    tar -xzf "$ROOT/.tools/uv.tgz" -C "$ROOT/.tools" --strip-components=1
    rm -f "$ROOT/.tools/uv.tgz"
    UV="$ROOT/.tools/uv"
fi
"$UV" --version

# --- Python 3.12 y entorno ---------------------------------------------------
echo; echo "== Python 3.12"
PYSPEC=3.12
if ! "$UV" python install 3.12; then
    # discos sin enlaces (exFAT/FAT32): Python se descarga pero falla el enlace
    PYHOME="$(ls -d "$ROOT"/.python/cpython-3.12* 2>/dev/null | sort -r | head -1)"
    if [ -n "$PYHOME" ] && [ -x "$PYHOME/bin/python3.12" ]; then
        echo "  aviso: el disco no admite enlaces; se usa $(basename "$PYHOME")"
        PYSPEC="$PYHOME/bin/python3.12"
    else
        echo "No se pudo instalar Python 3.12"; exit 1
    fi
fi
if [ ! -x .venv/bin/python ]; then
    echo "  creando entorno .venv"
    "$UV" venv .venv --python "$PYSPEC" --seed
else
    echo "  entorno .venv ya existe"
fi

# --- paquetes, modelos, diagnóstico, iniciar.sh ------------------------------
.venv/bin/python tools/install.py --uv "$UV" $CPU

# --- accesos directos --------------------------------------------------------
echo; echo "== Acceso directo"
APPS="$HOME/.local/share/applications"
if [ "$(uname -s)" = Linux ]; then
    mkdir -p "$APPS"
    cat > "$APPS/sistema-escaleras.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Sistema Escaleras
Comment=Arranca el sistema de prevención de caídas y abre el panel
Exec="$ROOT/iniciar.sh"
Path=$ROOT
Terminal=true
Icon=camera-web
Categories=Utility;
EOF
    chmod +x "$APPS/sistema-escaleras.desktop"
    echo "  menú de aplicaciones: Sistema Escaleras"
    DESK="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
    if [ -d "$DESK" ]; then
        cp "$APPS/sistema-escaleras.desktop" "$DESK/" && chmod +x "$DESK/sistema-escaleras.desktop"
        echo "  escritorio: $DESK/sistema-escaleras.desktop"
    fi
fi
# WSL: acceso directo en el Escritorio de WINDOWS que arranca dentro de Ubuntu
if grep -qi microsoft /proc/version 2>/dev/null && command -v powershell.exe >/dev/null; then
    DISTRO="${WSL_DISTRO_NAME:-Ubuntu}"
    powershell.exe -NoProfile -Command "
\$d = [Environment]::GetFolderPath('Desktop');
\$s = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path \$d 'Sistema Escaleras.lnk'));
\$s.TargetPath = 'C:\\Windows\\System32\\wsl.exe';
\$s.Arguments = '-d $DISTRO --cd \"$ROOT\" -- ./iniciar.sh';
\$s.Description = 'Arranca el sistema de prevencion de caidas (WSL) y abre el panel';
\$s.IconLocation = \"\$env:SystemRoot\\System32\\shell32.dll,137\";
\$s.Save(); Write-Host ('  Escritorio de Windows: ' + (Join-Path \$d 'Sistema Escaleras.lnk'))" \
    || echo "  (no se pudo crear el acceso directo de Windows)"
fi

echo
echo "================================================================"
echo " Instalación completa. Para arrancar: doble clic en 'Sistema"
echo " Escaleras' (Escritorio) o ./iniciar.sh"
echo "================================================================"

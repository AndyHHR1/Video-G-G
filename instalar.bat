@echo off
rem ===========================================================================
rem  Instalador para Windows: doble clic. Comprueba e instala solo lo que falta
rem  (Python 3.12, paquetes, modelos) dentro de esta carpeta y crea el
rem  arrancador "Sistema Escaleras" en el Escritorio.
rem  Opcion: instalar.bat --cpu   (forzar PyTorch sin GPU)
rem ===========================================================================
cd /d "%~dp0"
set "EXTRA="
if /i "%~1"=="--cpu" set "EXTRA=-cpu"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\instalar.ps1" %EXTRA%
if errorlevel 1 (
  echo.
  echo La instalacion no se completo. Revisa los mensajes de arriba.
)
echo.
pause

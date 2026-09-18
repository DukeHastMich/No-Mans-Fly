@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run SETUP_AND_RUN.bat once first.
  pause
  goto :eof
)
".venv\Scripts\python.exe" -c "import numpy, scipy, numba" >nul 2>nul || goto :fail
".venv\Scripts\python.exe" validate_existing_live.py >nul 2>nul
if errorlevel 1 (
  ".venv\Scripts\python.exe" prepare_full_baseline.py || goto :fail
  ".venv\Scripts\python.exe" validate_specimen003_install.py || goto :fail
)
echo.
echo Specimen-003 will run until Ctrl+C.
echo It checkpoints every 30 simulated seconds and once more on exit.
echo One simulated clock is used by default: 0.10 world s = 100 neural ms.
".venv\Scripts\python.exe" specimen003_desktop.py --headless --steps 0 --world-dt 0.10 --autosave-sim-seconds 30 --threads auto
goto :eof
:fail
echo Preparation/validation failed. No live state was reset.
pause

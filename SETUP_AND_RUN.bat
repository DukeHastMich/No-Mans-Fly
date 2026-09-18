@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem No Man's Fly now owns its startup checks in the graphical splash.  This
rem bootstrap only finds Python, creates the local venv if needed, and launches it.

set "PY_CMD="
where py >nul 2>nul
if not errorlevel 1 (
  py -3 -c "import sys; assert sys.version_info.major == 3" >nul 2>nul
  if not errorlevel 1 set "PY_CMD=py -3"
)
if not defined PY_CMD (
  where python >nul 2>nul
  if not errorlevel 1 (
    python -c "import sys; assert sys.version_info.major == 3" >nul 2>nul
    if not errorlevel 1 set "PY_CMD=python"
  )
)
if not defined PY_CMD (
  echo Python 3 was not found. Install 64-bit Python 3.10+ and try again.
  pause
  exit /b 10
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating No Man's Fly Python environment...
  %PY_CMD% -m venv .venv
  if errorlevel 1 (
    echo Could not create .venv
    pause
    exit /b 11
  )
)

if exist ".venv\Scripts\pythonw.exe" (
  start "No Man's Fly" /wait ".venv\Scripts\pythonw.exe" "no_mans_fly_launcher.py"
) else (
  ".venv\Scripts\python.exe" "no_mans_fly_launcher.py"
)
exit /b %errorlevel%

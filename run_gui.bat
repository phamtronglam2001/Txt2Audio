@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Virtual environment not found. Running: uv sync
  uv sync
  if errorlevel 1 (
    echo [ERROR] uv sync failed.
    pause
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Could not activate .venv
  pause
  exit /b 1
)

python -m txt2audio.app
set ERR=%ERRORLEVEL%

if not "%ERR%"=="0" (
  echo [ERROR] App exited with code %ERR%
  pause
)

exit /b %ERR%

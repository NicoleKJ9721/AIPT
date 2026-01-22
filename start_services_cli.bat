@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo   AIPT - AI Industrial Platform Tool
echo   Starting Services...
echo ===================================================

set "ROOT=%~dp0"

:: Select backend Python (prefer repo-local venv, then legacy conda env, fallback to PATH python)
set "BACKEND_PYTHON="
set "VENV_PYTHON=%ROOT%.venv\Scripts\python.exe"
set "PREFERRED_PYTHON=D:\APPS\Environments\Anaconda3\envs\vape_gpu\python.exe"

if exist "!VENV_PYTHON!" (
    set "BACKEND_PYTHON=!VENV_PYTHON!"
    echo [INFO] Using backend python (venv): !BACKEND_PYTHON!
) else if exist "!PREFERRED_PYTHON!" (
    set "BACKEND_PYTHON=!PREFERRED_PYTHON!"
    echo [INFO] Using backend python (conda): !BACKEND_PYTHON!
) else (
    echo [WARN] Preferred backend python not found.
    echo [WARN] Falling back to: python
    set "BACKEND_PYTHON=python"
)

:: Check backend Python
"!BACKEND_PYTHON!" --version >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] Backend Python is not available.
    echo         Please install Python or fix BACKEND_PYTHON in start_services_cli.bat
    pause
    exit /b 1
)

:: Check for Node.js
node --version >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] Node.js is not installed or not in PATH.
    pause
    exit /b 1
)

:: Create a logs directory if it doesn't exist
if not exist logs mkdir logs

echo.
echo [0/4] Checking backend dependencies...
"!BACKEND_PYTHON!" -c "import fastapi, uvicorn, sqlalchemy; import PIL, numpy; import torch, ultralytics" >nul 2>&1
if !errorlevel! neq 0 (
    echo    - Missing backend dependencies
    echo    - Installing ^(this is a one-time setup^)...
    "!BACKEND_PYTHON!" -m pip --version >nul 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] pip is not available in backend python environment.
        pause
        exit /b 1
    )
    "!BACKEND_PYTHON!" -m pip install -r "%ROOT%backend\requirements.txt" > logs\backend_install.log 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to install backend dependencies.
        echo         See logs\backend_install.log for details.
        pause
        exit /b 1
    )
    echo    - Backend dependencies installed.
) else (
    echo    - Backend dependencies OK.
)

echo.
echo [INFO] Ensuring ports are free (8000 backend, 5173 frontend)...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ports=@(8000,5173); foreach($p in $ports){ try { $pid=(Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess); if($pid){ Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue } } catch {} }" >nul 2>&1

echo.
echo [1/4] Starting Backend Service (Port 8000)...
:: NOTE: Use --reload-dir to avoid uvicorn scanning unrelated/broken folders under repo root.
start "AIPT Backend" /min cmd /c "cd /d %~dp0 && ""!BACKEND_PYTHON!"" -m uvicorn backend.main:app --reload --reload-dir backend --host 127.0.0.1 --port 8000 > logs\backend.log 2>&1"
echo    - Backend process launched.

echo.
echo Waiting for Backend to be ready (http://127.0.0.1:8000/health, /datasets)...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ok=$false; for($i=0;$i -lt 60;$i++){ try { Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2 | Out-Null; Invoke-RestMethod -Uri 'http://127.0.0.1:8000/datasets' -TimeoutSec 2 | Out-Null; $ok=$true; break } catch { Start-Sleep -Seconds 1 } }; if($ok){ exit 0 } else { exit 1 }" >nul 2>&1
if !errorlevel! equ 0 (
    echo    - Backend is ready.
) else (
    echo [WARN] Backend is not ready yet. Check logs\backend.log
)

echo.
echo [2/4] Checking frontend dependencies...
if not exist "%ROOT%frontend\node_modules" (
    echo    - node_modules not found. Installing frontend dependencies...
    pushd "%ROOT%frontend"
    npm install > "%ROOT%logs\frontend_install.log" 2>&1
    if !errorlevel! neq 0 (
        popd
        echo [ERROR] Failed to install frontend dependencies.
        echo         See logs\frontend_install.log for details.
        pause
        exit /b 1
    )
    popd
    echo    - Frontend dependencies installed.
) else (
    echo    - Frontend dependencies OK.
)

echo.
echo [3/4] Starting Frontend Service (Port 5173)...
start "AIPT Frontend" /min cmd /c "cd /d %~dp0\frontend && npm run dev > ..\logs\frontend.log 2>&1"
echo    - Frontend process launched.

echo.
echo Waiting for Frontend to be ready (http://localhost:5173)...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ok=$false; for($i=0;$i -lt 60;$i++){ try { Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:5173' -TimeoutSec 2 | Out-Null; $ok=$true; break } catch { Start-Sleep -Seconds 1 } }; if($ok){ exit 0 } else { exit 1 }" >nul 2>&1
if !errorlevel! equ 0 (
    echo    - Frontend is ready.
) else (
    echo [WARN] Frontend is not ready yet. Check logs\frontend.log
)

echo.
echo [4/4] Opening Dashboard...
start http://localhost:5173

echo.
echo ===================================================
echo   All services are running in background.
echo   - Backend: http://localhost:8000/docs
echo   - Frontend: http://localhost:5173
echo.
echo   Check 'logs' directory for detailed output.
echo   Press any key to stop all services...
echo ===================================================
pause >nul

echo.
echo Stopping services (ports 8000/5173)...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ports=@(8000,5173); foreach($p in $ports){ try { $pid=(Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess); if($pid){ Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue } } catch {} }" >nul 2>&1
echo Services stopped.

@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
set "PORT=8501"

if not exist "%PY%" (
    echo.
    echo [ERROR] venv not found: %PY%
    echo Please run  uv sync  in this folder first.
    echo.
    pause
    exit /b 1
)

netstat -ano | findstr ":%PORT%" | findstr "LISTENING" >nul
if %errorlevel%==0 (
    echo.
    echo [WARN] Port %PORT% is already in use.
    echo        Another instance may be running.
    echo        Close that window first, or just open:
    echo        http://localhost:%PORT%
    echo.
    pause
    exit /b 1
)

echo.
echo   Starting RAG Web UI ... please wait a few seconds.
echo   Browser will open automatically: http://localhost:%PORT%
echo   Close this window to stop the server.
echo.

rem Open the browser AFTER ~8s so the server is ready (avoids a blank page)
start "" /b cmd /c "ping -n 9 127.0.0.1 >nul & start http://localhost:%PORT%"

rem fileWatcherType none: disable the file watcher, which otherwise spams
rem tracebacks while scanning the transformers package.
"%PY%" -m streamlit run web.py --server.port %PORT% --server.headless true --server.fileWatcherType none

echo.
echo   Server stopped.
pause

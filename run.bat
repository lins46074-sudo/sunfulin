@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo.
    echo [ERROR] venv not found: %PY%
    echo Please run  uv sync  in this folder first.
    echo.
    pause
    exit /b 1
)

echo Starting RAG agent, please wait a few seconds...
echo.

"%PY%" chat.py

echo.
pause

@echo off
setlocal
set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

REM Zero-C-Drive Isolation: keep all caches & temp inside project directory
set "NGLAI_HOME=%ROOT_DIR%"
set "HF_HOME=%ROOT_DIR%\.cache\hf"
set "PIP_CACHE_DIR=%ROOT_DIR%\.cache\pip"
set "TMP=%ROOT_DIR%\.cache\tmp"
set "TEMP=%ROOT_DIR%\.cache\tmp"

if not exist "%ROOT_DIR%\.venv\Scripts\python.exe" (
    echo [New-Gen-Local-AI] Criando ambiente virtual isolado em "%ROOT_DIR%\.venv"...
    if not exist "%TMP%" mkdir "%TMP%"
    python -m venv "%ROOT_DIR%\.venv"
    "%ROOT_DIR%\.venv\Scripts\pip.exe" install -e "%ROOT_DIR%"
)

"%ROOT_DIR%\.venv\Scripts\python.exe" -m nglai.cli %*

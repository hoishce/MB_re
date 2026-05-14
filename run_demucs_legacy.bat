@echo off
setlocal

cd /d "%~dp0"

if defined MOCKINGBIRD_DEMUCS_LEGACY_VENV (
    set "VENV=%MOCKINGBIRD_DEMUCS_LEGACY_VENV%"
) else (
    set "VENV=%~dp0.venv_demucs_legacy"
)
set "PYTHON_EXE=%VENV%\Scripts\python.exe"

set "KMP_DUPLICATE_LIB_OK=TRUE"
set "PYTHONNOUSERSITE=1"
set "PATH=%VENV%;%VENV%\Scripts;%PATH%"

echo [Demucs legacy] Using Python:
if not exist "%PYTHON_EXE%" (
    echo Python executable not found: %PYTHON_EXE%
    echo Create the environment or set MOCKINGBIRD_DEMUCS_LEGACY_VENV.
    pause
    exit /b 1
)
"%PYTHON_EXE%" -c "import sys, numpy as np; print(sys.executable); print('numpy=', np.__version__); import torch; print('torch=', torch.__version__, 'cuda=', torch.version.cuda, 'available=', torch.cuda.is_available())"
if errorlevel 1 (
    echo Environment check failed.
    pause
    exit /b 1
)

if "%~1"=="" (
    echo Usage: run_demucs_legacy.bat path\to\script.py [args...]
    echo This environment is only for Demucs separation.
) else (
    "%PYTHON_EXE%" %*
)

pause

@echo off
setlocal

cd /d "%~dp0"

if defined MOCKINGBIRD_ADVANCED_VENV (
    set "VENV=%MOCKINGBIRD_ADVANCED_VENV%"
) else (
    set "VENV=%~dp0.venv_advanced"
)
set "PYTHON_EXE=%VENV%\Scripts\python.exe"

set "MOCKINGBIRD_PIPELINE_PROFILE=advanced"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "PYTHONNOUSERSITE=1"
set "PATH=%VENV%;%VENV%\Scripts;%PATH%"

echo [MockingBird advanced] Using Python:
if not exist "%PYTHON_EXE%" (
    echo Python executable not found: %PYTHON_EXE%
    echo Create the environment or set MOCKINGBIRD_ADVANCED_VENV.
    pause
    exit /b 1
)
"%PYTHON_EXE%" -c "import sys, numpy as np, torch; print(sys.executable); print('numpy=', np.__version__); print('torch=', torch.__version__, 'cuda=', torch.version.cuda, 'available=', torch.cuda.is_available())"
if errorlevel 1 (
    echo Environment check failed.
    pause
    exit /b 1
)

if "%~1"=="" (
    echo Usage: run_mockingbird_advanced.bat path\to\script.py [args...]
    echo Example: run_mockingbird_advanced.bat tools\run_advanced_batch.py --help
) else (
    "%PYTHON_EXE%" %*
)

pause

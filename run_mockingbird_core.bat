@echo off
setlocal

cd /d "%~dp0"

if defined MOCKINGBIRD_CORE_PYTHON (
    set "PYTHON_EXE=%MOCKINGBIRD_CORE_PYTHON%"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

set "MOCKINGBIRD_PIPELINE_PROFILE=core"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "PYTHONNOUSERSITE=1"
if exist "%~dp0.venv\Scripts" set "PATH=%~dp0.venv;%~dp0.venv\Scripts;%PATH%"

echo [MockingBird core] Using Python:
"%PYTHON_EXE%" -c "import sys, numpy as np, torch; print(sys.executable); print('numpy=', np.__version__); print('torch=', torch.__version__, 'cuda=', torch.version.cuda, 'available=', torch.cuda.is_available())"
if errorlevel 1 (
    echo Environment check failed.
    pause
    exit /b 1
)

if "%~1"=="" (
    "%PYTHON_EXE%" "demo\pipeline.py"
) else (
    "%PYTHON_EXE%" %*
)

pause

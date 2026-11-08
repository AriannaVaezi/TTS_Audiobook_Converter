@echo off
echo ===================================================
echo     Setting up Virtual Environment for Audiobook TTS
echo ===================================================

where uv >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [*] Found 'uv'. Creating Python 3.11 virtual environment with pip...
    uv venv --seed --python 3.11 .venv
    echo [*] Installing requirements from requirements.txt via uv...
    uv pip install -r requirements.txt --index-strategy unsafe-best-match
) else (
    echo [*] 'uv' not found. Creating standard virtual environment via Python...
    python -m venv .venv
    echo [*] Activating environment...
    call .venv\Scripts\activate.bat
    echo [*] Installing requirements from requirements.txt...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
)

echo.
echo ===================================================
echo [✓] Environment setup completed successfully!
echo To activate in the future, run: .venv\Scripts\activate
echo ===================================================
pause

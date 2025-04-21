@echo off
setlocal enabledelayedexpansion

:: --- Configuration ---
set PYTHON_CMD=py -3
set VENV_DIR=venv
set REQUIREMENTS_FILE=requirements.txt
set APP_SCRIPT=app.py

echo --- Starting Batch Script ---

:: --- Sanity Checks ---
echo Checking for Python command: %PYTHON_CMD% by running --version...
%PYTHON_CMD% --version >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Failed to execute '%PYTHON_CMD% --version'.
    echo [ERROR] Please ensure Python 3 is installed and the 'py' launcher is available.
    goto :eof_pause
)
echo Python command (%PYTHON_CMD%) appears executable.

:: --- SKIPPING file existence checks (if not exist) due to unexpected errors ---
echo Assuming %REQUIREMENTS_FILE% and %APP_SCRIPT% exist in %cd%.

:: --- Environment Setup ---
echo Checking for virtual environment directory: %VENV_DIR%...
if not exist "%VENV_DIR%\" (
    echo Virtual environment '%VENV_DIR%' not found. Creating...
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment in '%VENV_DIR%'. Check permissions or Python installation.
        goto :eof_pause
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment '%VENV_DIR%' already exists. Skipping creation.
)
echo --- Pausing after venv check/creation ---
pause

:: --- Install Dependencies ---
echo Installing dependencies from %REQUIREMENTS_FILE%... Please wait, this might take a while.

:: --- Use python -m pip for the upgrade command ---
echo Upgrading pip using recommended method...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo [WARNING] Failed to upgrade pip using '%VENV_DIR%\Scripts\python.exe -m pip'. Continuing anyway...
) else (
    echo Pip upgrade successful or already up-to-date.
)

echo Installing requirements...
"%VENV_DIR%\Scripts\pip.exe" install -r "%REQUIREMENTS_FILE%"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies using '%VENV_DIR%\Scripts\pip.exe'.
    echo [ERROR] Potential Issues: '%REQUIREMENTS_FILE%' not found or invalid, network error, INSUFFICIENT DISK SPACE. See pip messages above.
    goto :eof_pause
)
echo Dependencies installed successfully.
echo --- Pausing after dependency installation ---
pause

:: --- Run Application ---
echo Starting Streamlit application (%APP_SCRIPT%)...
echo The application should open in your web browser. Close this window or press Ctrl+C here to stop the app.
"%VENV_DIR%\Scripts\streamlit.exe" run "%APP_SCRIPT%"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to run Streamlit command: '%VENV_DIR%\Scripts\streamlit.exe run %APP_SCRIPT%'.
    echo [ERROR] Potential Issues: Streamlit not installed correctly (check previous step), '%APP_SCRIPT%' not found or contains errors.
    goto :eof_pause
)

echo Application finished or stopped by user.

:eof_pause
echo --- Script finished. Press any key to close this window. ---
pause
goto :eof

:eof
endlocal
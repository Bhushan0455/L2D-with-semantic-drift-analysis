@echo off
echo ============================================================
echo   Starting L2D Interactive Testing UI
echo ============================================================
cd /d "%~dp0"

:: Check if Anaconda python exists on typical path, otherwise use default python
if exist "C:\Users\%USERNAME%\anaconda3\python.exe" (
    echo Using Anaconda Python...
    "C:\Users\%USERNAME%\anaconda3\python.exe" Phase6_interactive_app.py
) else (
    echo Using system Python...
    python Phase6_interactive_app.py
)

pause

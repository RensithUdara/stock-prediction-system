@echo off
echo ==========================================
echo    Stock Prediction Hub - Auto Updater
echo ==========================================
echo.

:menu
echo Please choose an option:
echo [1] Start Auto-Updater (Every 15 minutes)
echo [2] Run Single Update
echo [3] Stop Auto-Updater
echo [4] View Update Log
echo [5] View Auto-Updater Status
echo [6] Exit
echo.
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto start_updater
if "%choice%"=="2" goto single_update
if "%choice%"=="3" goto stop_updater
if "%choice%"=="4" goto view_log
if "%choice%"=="5" goto check_status
if "%choice%"=="6" goto exit

echo Invalid choice. Please try again.
goto menu

:start_updater
echo.
echo 🚀 Starting Auto-Updater...
echo This will update market data every 15 minutes and commit to Git.
echo Press Ctrl+C to stop the auto-updater.
echo.
python auto_updater.py
goto menu

:single_update
echo.
echo 🔄 Running single update...
python auto_updater.py --once
echo.
echo Update completed. Press any key to return to menu...
pause >nul
goto menu

:stop_updater
echo.
echo 🛑 Stopping Auto-Updater...
taskkill /f /im python.exe /fi "WINDOWTITLE eq auto_updater*" 2>nul
echo Auto-updater processes stopped.
echo.
pause
goto menu

:view_log
echo.
echo 📋 Recent Update Log:
echo ==========================================
if exist "update_log.txt" (
    type update_log.txt | more
) else (
    echo No log file found. Auto-updater hasn't run yet.
)
echo ==========================================
echo.
pause
goto menu

:check_status
echo.
echo 📊 Auto-Updater Status:
echo ==========================================
tasklist /fi "imagename eq python.exe" /fi "windowtitle eq auto_updater*" 2>nul | find "python.exe" >nul
if %errorlevel%==0 (
    echo ✅ Auto-updater is RUNNING
) else (
    echo ❌ Auto-updater is NOT RUNNING
)

if exist "data\latest_market_data.csv" (
    echo ✅ Market data file exists
    for %%i in ("data\latest_market_data.csv") do echo 📅 Last modified: %%~ti
) else (
    echo ❌ No market data file found
)

if exist "update_log.txt" (
    echo ✅ Update log exists
    echo 📝 Last log entry:
    for /f "delims=" %%i in ('type "update_log.txt" ^| tail -1 2^>nul') do echo    %%i
) else (
    echo ❌ No update log found
)
echo ==========================================
echo.
pause
goto menu

:exit
echo.
echo 👋 Thank you for using Stock Prediction Hub Auto-Updater!
echo.
exit /b 0

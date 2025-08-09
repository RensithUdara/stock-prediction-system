@echo off
title Stock Prediction Hub - Auto Updater Service
color 0A

echo ==========================================
echo   🚀 Stock Prediction Hub Auto-Updater   
echo ==========================================
echo.
echo Starting continuous auto-update service...
echo Updates every 15 minutes
echo Press Ctrl+C to stop
echo.

C:/Python312/python.exe auto_updater.py

pause

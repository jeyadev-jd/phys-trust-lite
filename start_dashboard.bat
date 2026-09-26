@echo off
rem PHYS-TRUST Lite dashboard: http://localhost:8600
cd /d "%~dp0"
start "" /min pythonw -m uvicorn server.app:app --port 8600
timeout /t 4 >nul
start "" http://localhost:8600

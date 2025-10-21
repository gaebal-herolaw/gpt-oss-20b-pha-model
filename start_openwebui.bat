@echo off
cd /d "%~dp0"
echo Starting OpenWebUI...
echo.
echo OpenWebUI will be available at: http://localhost:8080
echo.
echo Press Ctrl+C to stop the server
echo.
.venv\Scripts\open-webui.exe serve

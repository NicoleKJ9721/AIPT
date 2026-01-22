@echo off
setlocal

set "ROOT=%~dp0"

REM Launch the GUI launcher without leaving a visible console window.
REM Uses wscript (GUI host) to spawn PowerShell hidden and returns immediately.
wscript.exe "%ROOT%scripts\\launch_start_services_gui.vbs"

exit /b 0

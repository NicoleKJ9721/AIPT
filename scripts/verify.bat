@echo off
setlocal

set "ROOT=%~dp0.."
pushd "%ROOT%" >nul

powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\scripts\verify.ps1" %*
set "CODE=%ERRORLEVEL%"

popd >nul
exit /b %CODE%


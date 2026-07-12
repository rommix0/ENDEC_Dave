@echo off
REM Double-click launcher for the DaveMSX installer.
REM Runs the PowerShell installer next to this file; it self-elevates (UAC prompt).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-DaveMSX.ps1"

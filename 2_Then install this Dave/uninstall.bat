@echo off
REM Double-click launcher for the DaveMSX uninstaller.
REM Runs the PowerShell uninstaller next to this file; it self-elevates (UAC prompt).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Uninstall-DaveMSX.ps1"

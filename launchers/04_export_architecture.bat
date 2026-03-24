@echo off
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "scripts\export_architecture_diagram.py"
) else (
  python "scripts\export_architecture_diagram.py"
)

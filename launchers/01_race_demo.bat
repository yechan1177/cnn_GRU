@echo off
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "launchers\run_race_demo.py"
) else (
  python "launchers\run_race_demo.py"
)

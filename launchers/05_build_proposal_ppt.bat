@echo off
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "scripts\build_proposal_ppt.py"
) else (
  python "scripts\build_proposal_ppt.py"
)

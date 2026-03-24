@echo off
setlocal
cd /d %~dp0..

if not exist .venv\Scripts\python.exe (
  echo .venv\Scripts\python.exe 를 찾을 수 없습니다.
  exit /b 1
)

.venv\Scripts\python.exe -m vcp.tools.brake_review_ui %*


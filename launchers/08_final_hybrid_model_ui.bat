@echo off
cd /d C:\yolstm
call .\.venv\Scripts\activate.bat
python -m vcp.tools.final_hybrid_model_ui %*

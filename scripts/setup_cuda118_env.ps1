# CUDA 11.8 + RTX 학습 환경 세팅 스크립트
# 사용: powershell -ExecutionPolicy Bypass -File .\scripts\setup_cuda118_env.ps1

param(
  [string]$VenvPath = '.\.venv'
)

python -m venv $VenvPath
& "$VenvPath\Scripts\python.exe" -m pip install --upgrade pip
& "$VenvPath\Scripts\python.exe" -m pip install -r requirements.txt
& "$VenvPath\Scripts\python.exe" -m pip install -r requirements-train.txt
& "$VenvPath\Scripts\python.exe" -m pip install -e .
& "$VenvPath\Scripts\python.exe" -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

& "$VenvPath\Scripts\python.exe" -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_version', torch.version.cuda); print('gpu_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

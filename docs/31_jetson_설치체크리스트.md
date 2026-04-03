# 31. Jetson Orin Nano 설치 체크리스트

## 목적
이 문서는 Jetson Orin Nano에서 본 저장소를 실제로 배포 가능한 상태로 올리기 위한 설치 체크리스트다. 목적은 먼저 Jetson을 안정적인 개발 상태로 준비한 뒤, 저장소를 그대로 실행할 수 있는 기본 환경을 만드는 것이다.

## 권장 기준
- 보드: Jetson Orin Nano Developer Kit
- JetPack: 6.2.x 계열 권장
- Python: 3.10+
- 저장장치: microSD 또는 NVMe SSD
- 네트워크: 유선 LAN 또는 안정적인 Wi-Fi 권장

## 1. 하드웨어 준비
- [ ] Jetson Orin Nano 본체
- [ ] 정격 전원 어댑터
- [ ] microSD 카드 또는 NVMe SSD
- [ ] 모니터/키보드/마우스
- [ ] 네트워크 연결 환경

## 2. JetPack 설치
### 방법 A. microSD 이미지 사용
- [ ] NVIDIA 공식 JetPack SD Card Image 다운로드
- [ ] Balena Etcher 등으로 microSD에 이미지 기록
- [ ] Jetson에 삽입 후 부팅
- [ ] 최초 Ubuntu 초기 설정 완료

### 방법 B. SDK Manager 사용
- [ ] Ubuntu 호스트 PC 준비
- [ ] SDK Manager 설치
- [ ] Jetson을 복구 모드로 연결
- [ ] JetPack 6.2.x 플래시 완료

## 3. 시스템 버전 확인
부팅 후 아래 명령으로 기본 버전을 확인한다.

```bash
uname -a
cat /etc/os-release
python3 --version
nvcc --version
```

체크 항목:
- [ ] Ubuntu 22.04 계열 확인
- [ ] Python 3.10 이상 확인
- [ ] CUDA 설치 확인

## 4. 기본 패키지 설치
```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y git curl wget htop tmux ffmpeg python3-pip python3-venv libopenblas-dev build-essential pkg-config
```

체크 항목:
- [ ] 패키지 설치 성공
- [ ] `git --version` 확인
- [ ] `ffmpeg -version` 확인

## 5. PyTorch for Jetson 설치 전 준비
Jetson에서는 일반 x86용 PyTorch wheel 대신 JetPack 버전에 맞는 NVIDIA 제공 wheel을 사용해야 한다.

```bash
sudo apt-get -y update
sudo apt-get install -y python3-pip libopenblas-dev
python3 -m pip install --upgrade pip
python3 -m pip install numpy
```

체크 항목:
- [ ] pip 업그레이드 완료
- [ ] numpy 설치 완료

## 6. JetPack 버전에 맞는 PyTorch wheel 설치
공식 문서를 보고 현재 JetPack 버전에 맞는 wheel URL을 선택한 뒤 설치한다.

```bash
python3 -m pip install --no-cache <NVIDIA_JETSON_TORCH_WHEEL_URL>
```

설치 후 확인:
```bash
python3 -c "import torch; print(torch.__version__)"
```

체크 항목:
- [ ] torch import 성공
- [ ] CUDA 사용 가능 여부 확인

## 7. torchvision 및 OpenCV 확인
```bash
python3 -c "import cv2; print(cv2.__version__)"
python3 -c "import torch; import torchvision; print('torchvision ok')"
```

체크 항목:
- [ ] OpenCV import 성공
- [ ] torchvision import 성공

## 8. 전력/클럭 관리 도구 확인
```bash
sudo nvpmodel -q
sudo jetson_clocks --show
```

체크 항목:
- [ ] `nvpmodel` 사용 가능
- [ ] `jetson_clocks` 사용 가능
- [ ] `tegrastats` 실행 가능

## 9. 저장소 준비 전 최종 상태
아래 항목이 모두 되면 저장소 배포를 시작할 수 있다.

- [ ] JetPack 설치 완료
- [ ] Python 3.10+ 확인
- [ ] PyTorch for Jetson 설치 완료
- [ ] OpenCV 및 torchvision import 성공
- [ ] 전력/클럭 관리 도구 확인
- [ ] 기본 네트워크 연결 확인

## 참고 링크
- Jetson Orin Nano Developer Kit Software Setup
  - https://developer.nvidia.com/embedded/learn/jetson-orin-nano-devkit-user-guide/software_setup.html
- JetPack SDK
  - https://developer.nvidia.com/embedded/jetpack-sdk-622
- Installing PyTorch for Jetson Platform
  - https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html

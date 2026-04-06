# Spatial Encoder 간단 재현 가이드

`practice_spatial_encoder.py`는 `src/vcp/components/spatial.py`의 핵심 아이디어를 처음 보는 사람도 바로 실행해볼 수 있도록 단순화한 루트 실행 스크립트다.

## 목적
- YOLO 검출 결과가 어떻게 16차원 순간 특징 벡터로 변환되는지 빠르게 확인한다.
- 복잡한 UI나 시계열 모델 없이 `단일 프레임 -> detector -> feature vector` 흐름만 재현한다.
- 발표자료나 코드리뷰에서 사용할 예시 그림을 바로 생성한다.

## 입력
- 기본 입력 영상: `data/raw/videos/people_braking.mp4`
- 기본 YOLO 가중치: `models/checkpoints/yolo3cls_best.pt`
- 기본 프레임 인덱스: `180`

필요하면 스크립트 상단의 설정값만 바꾸면 된다.

## 실행 방법
```powershell
cd C:\yolstm
.\.venv\Scripts\python.exe practice_spatial_encoder.py
```

전역 Python 환경을 쓰는 경우에는 아래처럼 실행하면 된다.

```powershell
cd C:\yolstm
python practice_spatial_encoder.py
```

## 출력
- 콘솔에 16차원 특징 벡터 값 출력
- 루트 경로에 `practice_spatial_encoder_preview.png` 저장

미리보기 그림에는 아래 정보가 함께 포함된다.
- YOLO 검출 박스
- 중앙 ROI
- 대표 16차원 특징 값

## 코드 구조
이 스크립트는 아래 순서로 동작한다.

1. 영상에서 특정 프레임을 읽는다.
2. YOLO로 현재 프레임의 객체를 검출한다.
3. 검출 결과를 16차원 특징 벡터로 변환한다.
4. 박스, ROI, feature를 한 장의 그림으로 저장한다.

## 원본 구현과의 관계
- 교육용 단순화 버전: `practice_spatial_encoder.py`
- 실제 파이프라인 구현: `src/vcp/components/spatial.py`

연습 목적상 시간 변화량은 여기서 계산하지 않는다. 시간 흐름은 실제 파이프라인에서 `CNN-GRU`가 최근 8프레임 시퀀스를 보고 학습한다.

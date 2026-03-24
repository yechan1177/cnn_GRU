# 비전인식-맥락 연계형 온디바이스 파이프라인

실시간 영상에서 `person / vehicle / bike`를 검출하고, 16차원 의미 특징과 CNN-GRU 기반 맥락 모델로 주행 상황을 해석하는 프로젝트다. 최종 실행 파일은 루트의 `run_final_model.py`이며, mp4 또는 웹캠 입력을 바로 사용할 수 있다.

## 저장소에 포함한 핵심 구성
- `run_final_model.py`: 루트 단일 실행 파일
- `src/`: 모델, 데이터 변환, UI, 학습 코드
- `models/checkpoints/`: 실행에 필요한 최소 체크포인트
- `configs/hybrid_rule_params.json`: 하이브리드 룰 파라미터
- `launchers/`: Windows 실행 배치 파일
- `tests/`: 최소 회귀 테스트
- `docs/`, `tasks/`: 한국어 문서와 작업 이력

## 1. 설치
```powershell
cd C:\yolstm
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 2. 가장 빠른 실행
```powershell
cd C:\yolstm
.\.venv\Scripts\python.exe run_final_model.py
```

기본 동작:
- `data/raw/videos/people_braking.mp4`가 있으면 해당 영상을 사용한다.
- 영상이 없으면 자동으로 `webcam:0`으로 전환한다.

## 3. 입력 소스 바꾸기
`run_final_model.py` 상단 설정만 수정하면 된다.

```python
USE_WEBCAM = False
WEBCAM_INDEX = 0
VIDEO_SOURCE = str(ROOT_DIR / "data" / "raw" / "videos" / "people_braking.mp4")
```

- mp4 사용: `USE_WEBCAM = False`, `VIDEO_SOURCE` 수정
- 웹캠 사용: `USE_WEBCAM = True`, `WEBCAM_INDEX` 수정

## 4. 화면에 표시되는 정보
- YOLO 검출 박스
- `pure model`: 순수 CNN-GRU 예측
- `hybrid final`: 룰 게이트 적용 최종 태그
- `boundary`
- `brake_warning / hard_brake_risk / front_vehicle_follow` 확률
- `roi / center / looming / occlusion / motion_delta`

키 입력:
- `space`: 재생/일시정지
- `s`: 스크린샷 저장
- `q`: 종료

## 5. 보조 실행 명령
하이브리드 검수 UI:
```powershell
.\.venv\Scripts\python.exe -m vcp.tools.final_hybrid_model_ui
```

브레이크 구간 검수 UI:
```powershell
.\.venv\Scripts\python.exe -m vcp.tools.brake_review_ui
```

YOLO 3클래스 학습:
```powershell
.\.venv\Scripts\python.exe -m vcp.tools.train_yolo_nano --data configs/datasets/yolo3cls_merged.yaml --model models/pretrained/yolov8n.pt --epochs 150 --final-epochs 20 --imgsz 640 --batch 16 --device 0 --optimizer Adam --patience 20 --project experiments/exp_011_yolo3cls_training/runs --name yolov8n_3cls_from_pretrained
```

Temporal 학습:
```powershell
.\.venv\Scripts\python.exe -m vcp.tools.train_temporal_gru --dataset-dir data/processed/<dataset_dir> --epochs 24 --batch-size 256 --hidden-dim 96 --cnn-channels 24 --device cuda:0 --project experiments/temporal_runs --name temporal_run
```

## 6. 테스트
```powershell
cd C:\yolstm
.\.venv\Scripts\python.exe -m pytest -q
```

## 7. 저장소 정책
- 대용량 데이터셋, 실험 산출물, 영상 파일은 저장소에 포함하지 않는다.
- 실행에 필요한 최소 체크포인트만 `models/checkpoints/`에 포함한다.
- 문서와 코드 주석은 한국어 기준으로 유지한다.

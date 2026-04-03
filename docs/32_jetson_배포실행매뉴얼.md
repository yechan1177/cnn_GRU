# 32. Jetson 배포 실행 매뉴얼

## 목적
이 문서는 Jetson Orin Nano에서 본 저장소를 실제로 실행하고, 3개 비교 모델과 논문 재현 산출물을 확인하기 위한 명령 중심 매뉴얼이다. 설치 체크리스트를 완료한 뒤 이 문서를 순서대로 따라가면 된다.

## 1. 저장소 받기
```bash
cd ~
git clone -b codex/paper-repro-20260330 https://github.com/yechan1177/cnn_GRU.git
cd cnn_GRU
```

## 2. 가상환경 생성
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 3. 저장소 의존성 설치
주의: Jetson용 PyTorch를 먼저 설치한 뒤 아래 명령을 실행한다.

```bash
pip install -r requirements.txt
pip install -e .
```

## 4. 기본 import 점검
```bash
python -c "import torch; print(torch.__version__)"
python -c "import cv2; print(cv2.__version__)"
python -c "from ultralytics import YOLO; print('ultralytics ok')"
python -c "import vcp; print('vcp ok')"
```

## 5. 체크포인트 확인
아래 파일이 존재하는지 확인한다.

```bash
ls models/checkpoints
ls configs
```

필수 파일:
- `models/checkpoints/yolo3cls_best.pt`
- `models/checkpoints/temporal_shared_best.pt`
- `models/checkpoints/temporal_final_best.pt`
- `models/checkpoints/literature_temporal_best.pt`
- `configs/hybrid_rule_params.json`

## 6. 입력 영상 준비
논문 재현 및 기능 검증에 사용할 영상을 `data/raw/videos/` 아래에 배치한다.

예:
- `people_braking.mp4`
- `stopcar.mp4`
- `rainy_stopcar.mp4`
- `stopcar1.mp4`
- `stopcar2.mp4`
- `race.mp4`

## 7. 1차 기능 검증
### 7-1. YOLO + 순간 규칙
```bash
python run_yolo_rule_model.py
```

확인 항목:
- YOLO 박스 표시 여부
- 순간 규칙 라벨 변화 여부
- 영상 종료까지 에러 없이 동작하는지

### 7-2. CNN-GRU only
```bash
python run_baseline_cnn_gru_model.py
```

확인 항목:
- 최근 8프레임 시퀀스 누적 여부
- pure model 예측 표시 여부
- 체크포인트 로드 성공 여부

### 7-3. CNN-GRU + 규칙 보정
```bash
python run_final_model.py
```

확인 항목:
- pure model / hybrid final 동시 표시 여부
- rule reason 표시 여부
- 이벤트 전환 시 최종 라벨 변화 여부

## 8. 세 모델 동시 비교 실행
```bash
python run_compare_three_models.py
```

생성 산출물:
- `artifacts/comparisons/<video_stem>_three_model_compare/summary.json`
- `artifacts/comparisons/<video_stem>_three_model_compare/frame_comparison.csv`
- `artifacts/comparisons/<video_stem>_three_model_compare/comparison_plot.png`
- `artifacts/comparisons/<video_stem>_three_model_compare/event_records.json`
- `artifacts/comparisons/<video_stem>_three_model_compare/screenshots/`
- `artifacts/comparisons/<video_stem>_three_model_compare/clips/`

## 9. 논문용 그래프 재생성
```bash
python scripts/export_three_class_paper_bar_chart.py
python scripts/export_three_model_class_count_graph.py
python scripts/export_manual_hard_brake_range_summary.py
python scripts/export_merged_brake_event_metric_comparison.py
```

생성 파일:
- `three_class_paper_bar_chart.png`
- `three_model_class_count_comparison.png`
- `manual_hard_brake_range_summary.png`
- `merged_brake_class_count_comparison.png`
- `paper_metric_comparison_with_event_score.png`

## 10. 성능 측정 기본 절차
### 10-1. tegrastats 로그 저장
터미널 1:
```bash
sudo tegrastats --interval 1000 > tegrastats_final_model.log
```

터미널 2:
```bash
python run_final_model.py
```

실행이 끝나면 `Ctrl+C`로 로그를 종료한다.

### 10-2. 전력 모드 확인
```bash
sudo nvpmodel -q
sudo jetson_clocks --show
```

필요 시 최대 성능 모드로 설정한다.
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

주의: 실제 모드 번호는 장치와 JetPack 버전에 따라 다를 수 있으므로 먼저 `nvpmodel -q --verbose`로 확인한다.

## 11. 성능 검증에 바로 쓸 항목
`summary.json`에서 아래 항목을 확인한다.
- `A_det`
- `T_inf_rule_ms`
- `T_inf_temporal_ms`
- `T_inf_hybrid_ms`
- `N_param_rule`
- `N_param_temporal`
- `N_param_hybrid`

사용자 수동 판정으로 계산할 식:
\[
P_{ctx} = \frac{N_{match}}{N_{total}}
\]

## 12. 배포 단계 권장 순서
1. Jetson 기본 설치 완료
2. 저장소 복제 및 의존성 설치
3. 3개 개별 실행기 기능 검증
4. 세 모델 동시 비교 실행
5. 논문용 그래프 재생성
6. tegrastats 기반 자원 사용량 측정
7. 전력 모드별 비교
8. 필요 시 TensorRT/최적화 단계로 확장

## 13. 문제 발생 시 우선 점검 항목
- Python 버전이 3.10 이상인지
- JetPack 버전에 맞는 PyTorch wheel을 설치했는지
- `import vcp`가 되는지
- 체크포인트 경로가 맞는지
- 입력 영상 파일명이 저장소 기준과 일치하는지
- OpenCV 비디오 디코딩이 정상 동작하는지

## 관련 문서
- `docs/31_jetson_설치체크리스트.md`
- `docs/29_논문재현가이드.md`
- `docs/28_세모델정의재구성.md`
- `docs/04_배포전략.md`

# 비전인식-맥락 연계형 온디바이스 파이프라인

실시간 주행 영상을 입력으로 받아 객체를 검출하고, 의미 기반 특징 벡터와 CNN-GRU 시계열 모델을 통해 주행 맥락을 해석하는 저장소다. 기본 목적은 **온디바이스에서 바로 실행 가능한 경량 파이프라인**을 제공하고, 이후 멀티모달 데이터셋 자동 구축으로 확장 가능한 기반을 만드는 것이다.

![파이프라인 개요](docs/assets/readme_pipeline_overview.svg)

## 개요
이 프로젝트는 아래 흐름으로 동작한다.

1. `YOLOv8n`으로 `person / vehicle / bike`를 검출한다.
2. 검출 결과를 `16차원 의미 특징 벡터`로 변환한다.
3. 최근 `8프레임`을 `CNN-GRU`에 입력해 맥락을 예측한다.
4. 하이브리드 룰 게이트로 최종 태그를 보정한다.
5. UI에서 순수 모델 예측과 최종 하이브리드 결과를 함께 확인한다.

## 2026-10 VLA 연계 논문(v2): 수집 시점 큐레이션 CARE

차량·이동로봇이 저장 예산 안에서 무엇을 남겨야 VLA 정책이 희소 위험 상황에서 실패하지 않는지를, **선별 데이터로 학습한 소형 VLA 정책의 폐루프 성능**으로 검증했다.

- 원고: [paper/manuscript_vla_ko.md](paper/manuscript_vla_ko.md) — 자동 생성, 원천은 `paper/sections_vla/`
- 계획·인터페이스: [docs/28](docs/28_VLA_논문_재설계_계획.md), [docs/28a](docs/28a_VLA_모듈_인터페이스_계약.md)
- 모듈 문서: [docs/29 시뮬레이터·폐루프](docs/29_VLA_시뮬레이터_폐루프.md), [docs/30 큐레이션 방법](docs/30_큐레이션_방법_정의.md), [docs/31 VLA-lite 정책](docs/31_VLA_lite_정책.md)
- 결과 요약: `experiments/exp_110_vla_curation/summary/`(표 `tables/vla_*.md`, 그림 `figures/`, 수치 `vla_stats.json`)
- 재현
  - 실험: `bash scripts/run_vla_curation.sh` — 단계별 캐시, CPU 4코어 기준 약 15시간
  - 집계: `python -m vcp.experiments.vla_report`
  - 원고: `python scripts/build_vla_paper.py`

### 주요 결과

시뮬레이터 결과는 테스트 147 에피소드, 시드 3개 기준이다.

| 결과 | 내용 |
|---|---|
| 주행, 저장 예산 2% | CARE 폐루프 성공률 0.751, 무작위 0.644. 차이 +0.107 [95% CI +0.057, +0.154] |
| 예산별 이득 | 1%에서도 유의한 이득. 5%·10%에서는 차이 없음 |
| 위험 편향 붕괴 | 위험 표본만 늘리는 선별은 모든 예산에서 성공률 0.26 이하. 대상: 감속 트리거, 맥락 이벤트, GT 오라클, 정책 손실 기반 오프라인 선별 |
| 개루프 지표 | 위험 구간 개루프 오차는 폐루프 성공률과 상관 없음 |
| 실내 이동로봇(AMR) | CARE가 무작위보다 낮음(재튜닝 후에도) |
| 실주행 영상(개루프) | 방법 간 차이를 측정하지 못함. 위험 편향 트레이드오프는 재현됨 |
| 언어 절제 | 초기 속도로 지시 정보가 새어 나가 언어 효과를 엄밀히 측정하지 못함(한계로 보고) |

## 2026-10 개정 요약
2026-03 제출본을 재검토해 평가 누수·라벨 순환·FPS 의존 특징 문제를 확인하고, 아래 항목을 추가했다. 상세는 [단계별 계획](docs/25_연구고도화_단계별계획.md), [실험 프로토콜](docs/27_실험프로토콜_자동실험.md), [피지컬 AI/VLA 연계](docs/26_피지컬AI_로봇_VLA_연계.md), [자동 실험 결과](experiments/exp_100_paper_suite/summary/RESULTS.md), [논문 원고](paper/manuscript_ko.md)를 참고한다.

| 항목 | 내용 |
|---|---|
| 특징 v2 | IoU 추적 + Δt 정규화 + 박스 크기 변화율 기반 역 TTC (FPS 불변) |
| 합성 데이터 | 물리 기반 주행/실내 이동로봇 시나리오, GT 물리량 라벨(특징과 독립) |
| 실데이터 | comma.ai speedchallenge 실주행 영상 + 속도 센서 라벨 |
| 자동 실험 | 에피소드/시간 블록 분할, 검증셋 전용 튜닝, 다중 시드, 이벤트 지표, CI |
| VLA 연계 | 행동 head, 한/영 맥락 서술, 예산 기반 큐레이션, LeRobot v2 구조 내보내기 |
| 클라우드 실행 | `--device auto`, `--no-display` headless 실행 |

### 주요 재평가 결과 (자동 실험, 상세: [RESULTS.md](experiments/exp_100_paper_suite/summary/RESULTS.md))
- 성능을 좌우한 것은 특징 설계다: 같은 구조에서 v1→v2로 바꾸면 macro-F1이 합성 0.506→0.580, 실주행(10-fold) 0.302→0.442.
- 의미 기반 채널 그룹과 하이브리드 룰 게이트는 macro-F1 개선을 보이지 않았다(2026-03 주장 수정).
- v1+v2 결합 + 창 16: 합성 최고 macro-F1 0.650, 실주행 최고 제동 AUROC 0.782.
- 저장 예산 20%에서 모델 기반 큐레이션이 제동 프레임 46%를 회수(무작위 20%).
- 수치는 GPU 없는 클라우드 CPU에서 측정했으며 Jetson Orin Nano는 미측정.

### 클라우드/서버(화면 없음)에서 실행
```bash
python -m pip install -e . && python -m pip install onnx onnxruntime pyarrow matplotlib koreanize-matplotlib
python run_final_model.py --source <영상.mp4> --no-display --device auto \
    --save-video artifacts/run.mp4 --output-jsonl artifacts/run.jsonl
```

### 논문 실험 전체 자동 실행
```bash
bash scripts/run_paper_suite.sh          # 공개 데이터 다운로드 → 검출 → 합성 데이터 생성 → 실험 → 리포트 → 원고
bash scripts/run_paper_suite.sh --quick  # 스모크 테스트
```

## 핵심 기능
| 기능 | 설명 |
|---|---|
| 실시간 추론 | mp4 또는 웹캠을 입력으로 받아 즉시 동작 |
| 경량 detector | YOLOv8n 기반 3클래스 검출 |
| 맥락 인식 | 16차원 의미 특징 + CNN-GRU |
| 하이브리드 보정 | boundary, ROI, looming, motion 기반 최종 보정 |
| 검수 루프 | 브레이크 구간 검수 UI 제공 |
| GitHub 실행성 | 상대경로 기반 실행, 최소 체크포인트 포함 |

## 저장소 구성
| 경로 | 용도 |
|---|---|
| `run_final_model.py` | 가장 빠른 단일 실행 파일 |
| `src/vcp/` | 파이프라인 본체 코드 |
| `models/checkpoints/` | 실행용 최소 체크포인트 |
| `configs/` | 런타임/학습 설정 |
| `launchers/` | Windows 배치 실행 파일 |
| `data/annotations/context/` | 수동 맥락 라벨 원본 |
| `tests/` | 회귀 테스트 |
| `docs/` | 한국어 설계 문서 |
| `tasks/` | 작업 이력 및 체크리스트 |
| `src/vcp/features/` | 검출 박스 → 의미 특징(v1 호환, v2 제안) |
| `src/vcp/sim/` | 물리 기반 합성 시나리오(주행/AMR) |
| `src/vcp/experiments/` | 자동 실험 스위트, 지표, 리포트 |
| `src/vcp/vla/` | VLA 연계(언어 서술, 큐레이션, 내보내기) |
| `experiments/exp_100_paper_suite/summary/` | 자동 실험 결과(추적) |
| `paper/` | 논문 원고(자동 생성) |

## 빠른 시작
### 1. 설치
```powershell
cd C:\yolstm
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 2. 바로 실행
```powershell
cd C:\yolstm
.\.venv\Scripts\python.exe run_final_model.py
```

기본 동작은 다음과 같다.
- `data/raw/videos/people_braking.mp4`가 있으면 해당 영상을 사용한다.
- 영상이 없으면 자동으로 `webcam:0`으로 전환한다(`--no-display` 모드에서는 오류로 종료).
- `--source`, `--device`, `--conf` 등 명령행 인자로 상단 설정을 덮어쓸 수 있다.

## 입력 소스 변경
`run_final_model.py`의 `--source` 인자(또는 상단 `DEFAULT_SOURCE`)로 지정한다.

| 사용 방식 | 설정 |
|---|---|
| mp4 파일 | `--source path/to/video.mp4` |
| 웹캠 | `--source webcam:0` |

## 화면에서 확인할 수 있는 정보
- YOLO 검출 박스
- `pure model`: 순수 CNN-GRU 예측
- `hybrid final`: 룰 게이트 적용 최종 태그
- `boundary`
- `brake_warning / hard_brake_risk / front_vehicle_follow` 확률
- `roi / center / looming / occlusion / motion_delta`

키 입력:
- `space`: 재생/일시정지
- `s`: 현재 화면 저장
- `q`: 종료

## 보조 실행 명령
### 최종 하이브리드 UI
```powershell
.\.venv\Scripts\python.exe -m vcp.tools.final_hybrid_model_ui
```

### 브레이크 구간 검수 UI
```powershell
.\.venv\Scripts\python.exe -m vcp.tools.brake_review_ui
```

### YOLO 3클래스 학습
```powershell
.\.venv\Scripts\python.exe -m vcp.tools.train_yolo_nano --data configs/datasets/yolo3cls_merged.yaml --model models/pretrained/yolov8n.pt --epochs 150 --final-epochs 20 --imgsz 640 --batch 16 --device 0 --optimizer Adam --patience 20 --project experiments/exp_011_yolo3cls_training/runs --name yolov8n_3cls_from_pretrained
```

### Temporal 학습
```powershell
.\.venv\Scripts\python.exe -m vcp.tools.train_temporal_gru --dataset-dir data/processed/<dataset_dir> --epochs 24 --batch-size 256 --hidden-dim 96 --cnn-channels 24 --device cuda:0 --project experiments/temporal_runs --name temporal_run
```

## 테스트
```powershell
cd C:\yolstm
.\.venv\Scripts\python.exe -m pytest -q
```

## 포함된 실행용 체크포인트
| 파일 | 용도 |
|---|---|
| `models/checkpoints/yolo3cls_best.pt` | 3클래스 detector |
| `models/checkpoints/temporal_final_best.pt` | 최종 temporal 모델 |
| `models/checkpoints/literature_temporal_best.pt` | 비교용 문헌형 temporal 모델 |
| `configs/hybrid_rule_params.json` | 하이브리드 룰 파라미터 |

## 문서 인덱스
추천 순서로 읽으면 된다.

1. [프로젝트 개요](docs/00_프로젝트개요.md)
2. [시스템 아키텍처](docs/01_시스템아키텍처.md)
3. [모델 설계](docs/03_모델설계.md)
4. [실험 방법](docs/05_실험방법.md)
5. [브레이크 리뷰 UI](docs/17_브레이크리뷰UI.md)
6. [최종 하이브리드 모델 UI](docs/20_최종하이브리드모델UI.md)
7. [연구 고도화 단계별 계획](docs/25_연구고도화_단계별계획.md)
8. [피지컬 AI · 로봇 · VLA 연계](docs/26_피지컬AI_로봇_VLA_연계.md)
9. [실험 프로토콜과 자동 실험](docs/27_실험프로토콜_자동실험.md)

## 저장소 정책
- 대용량 데이터셋, 실험 산출물, 영상 파일은 저장소에 포함하지 않는다.
- 실행에 필요한 최소 체크포인트만 `models/checkpoints/`에 포함한다.
- 문서와 코드 주석은 한국어 기준으로 유지한다.
- 학습 경로와 배포 경로는 분리 설계를 유지한다.

## 변경 이력
최근 변경 내용은 [CHANGELOG.md](CHANGELOG.md)에서 관리한다.

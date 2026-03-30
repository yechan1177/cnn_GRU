# 규칙 기반 pseudo-label 시계열 데이터셋

## 목적
- 영상 구간 이름을 그대로 외우는 방식이 아니라, 16차원 특징벡터 상태를 기준으로 GRU를 학습하기 위한 데이터셋을 생성한다.
- 각 프레임을 `normal_drive`, `front_vehicle_follow`, `brake_warning`, `hard_brake_risk`, `post_brake_recovery`, `dense_traffic` 중 하나로 자동 라벨링한다.

## 핵심 원리
- 최근 baseline window의 평균 특징값을 현재 프레임과 비교한다.
- `roi_risk`, `center_closeness`, `looming_score`, `motion_delta`, `mean_area`의 급격한 변화가 나타나면 brake 계열로 승격한다.
- brake가 한 번 시작되면 hold 프레임과 recovery 프레임을 둬서 상태가 너무 빠르게 흔들리지 않게 한다.

## 사용 명령
```powershell
cd C:\yolstm
.\.venv\Scripts\python.exe -m vcp.tools.build_rule_context_dataset --runs-root outputs/runs --output-root data/processed --dataset-name rule_context_feat16_v2 --run-glob *_yolo3cls_feat16_t045_demo_* --window-size 8 --val-ratio 0.2 --baseline-window 8
```

## 생성 결과
- `train.jsonl`
- `val.jsonl`
- `samples_all.jsonl`
- `label_map.json`
- `analysis.json`
- `dataset_manifest.json`

## 현재 기본 반영
- 기본 temporal checkpoint는 규칙 기반 pseudo-label 데이터셋으로 재학습한 모델로 교체되었다.
- 이전 수동 구간 중심 checkpoint는 `models/checkpoints/temporal_final_best_manual_segments_backup.pt`로 백업해 두었다.

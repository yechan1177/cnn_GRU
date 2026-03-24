# TASK_003: Temporal Encoder 고도화

## 목표
- GRU baseline을 학습/추론 분리 구조로 고도화한다.
- context/event boundary/uncertainty head를 분리해 학습 가능한 temporal 모듈을 구축한다.

## 체크리스트
- [x] temporal encoder 입력 윈도우 정규화/패딩 구현
- [x] context/event/uncertainty head 분리 구현 (`TemporalGRUNet`)
- [x] 학습/추론 그래프 분리 (`train_temporal_gru.py`, `GRUTemporalEncoderTorch`)
- [x] 체크포인트 로더 및 런타임 추론 연동
- [x] 실험 실행 및 비교 지표 기록
- [x] 실험 비교표 작성

## 산출물
- `src/vcp/components/temporal.py`
- `src/vcp/tools/train_temporal_gru.py`
- `experiments/exp_005_temporal_gru_training/runs/gru_e8_vehicle/`
- `artifacts/tables/performance/exp_005_temporal_gru_best_metrics.csv`

## 상태
- 완료

## 진행 메모
- 2026-03-18: `dataset_vehicle_validation_yolo6cls_20260318_150443` 기준 학습 완료
- 2026-03-18: best checkpoint를 runtime 파이프라인에 연결해 추론 동작 검증

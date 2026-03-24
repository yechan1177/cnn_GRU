# TASK_039 기본모델/문헌형/제안모델 맥락 벤치마크

## 목적
- [x] 동일 데이터셋과 동일 검증 분할 기준으로 비교군 3개를 실제 평가한다.
- [x] 기본모델, 문헌형 비교모델, 제안모델의 정의를 명확히 고정한다.
- [x] 정확도 외에 희소 브레이크 맥락을 반영하는 지표를 함께 산출한다.
- [x] 비교 표와 그래프를 논문/발표에 재사용 가능한 파일로 저장한다.
- [x] README 또는 docs에 비교 해석을 반영한다.

## 비교군 정의
- [x] 비교군 1: YOLO 출력 기반 규칙형 baseline
- [x] 비교군 2: 문헌형 단일채널 CNN-GRU baseline
- [x] 비교군 3: 제안 멀티채널 CNN-GRU

## 평가 데이터
- [x] `data/processed/dataset_manual_context_braking_expanded_feat16_t045_20260324_115253`
- [x] 기존 `train.jsonl`, `val.jsonl` 분할을 그대로 사용한다.

## 평가 지표
- [x] context accuracy
- [x] context macro F1
- [x] boundary F1
- [x] brake-critical recall (`brake_warning`, `hard_brake_risk`)

## 주의사항
- [x] 비교군 1은 detector-only에 가까운 규칙 기반이므로 정량 비교 시 정의를 명시한다.
- [x] 비교군 2는 실제 논문 코드를 그대로 재현한 것이 아니라 문헌 구조를 반영한 baseline임을 표시한다.
- [x] 같은 데이터 분할, 같은 지표, 같은 label map으로만 비교한다.
- [x] 검증되지 않은 우월성 문구는 쓰지 않는다.

## 구현 결과
- [x] benchmark 스크립트 추가
  - `src/vcp/tools/benchmark_context_models.py`
- [x] 실행 진입점 추가
  - `vcp-benchmark-context`
- [x] 실제 비교 실험 실행
  - `experiments/exp_016_context_model_benchmark/runs/context_model_benchmark_v1`
- [x] 루트 표/그래프 저장
  - `context_model_benchmark_actual_table.csv`
  - `context_model_benchmark_actual_table.md`
  - `context_model_benchmark_metrics.png`
  - `context_model_benchmark_manifest.json`

## 실제 결과
- [x] 기본 YOLO 규칙형 baseline
  - `context_acc=0.811667`
  - `context_macro_f1=0.149341`
  - `boundary_f1=0.000000`
  - `brake_critical_recall=0.000000`
- [x] 문헌형 단일채널 CNN-GRU
  - `context_acc=0.801667`
  - `context_macro_f1=0.220099`
  - `boundary_f1=0.041812`
  - `brake_critical_recall=0.222222`
- [x] 제안 멀티채널 CNN-GRU
  - `context_acc=0.818333`
  - `context_macro_f1=0.236823`
  - `boundary_f1=0.050633`
  - `brake_critical_recall=0.111111`

## 해석
- [x] 제안모델은 전체 accuracy, macro F1, boundary F1에서 가장 높다.
- [x] 문헌형 단일채널 CNN-GRU는 brake-critical recall이 더 높다.
- [x] 현재 제안모델은 전체 안정성은 더 낫지만 rare brake class 민감도는 추가 개선이 필요하다.

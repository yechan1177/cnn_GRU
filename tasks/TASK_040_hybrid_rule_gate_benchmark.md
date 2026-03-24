# TASK_040 제안모델 + 룰 게이트 하이브리드 비교

## 목적
- [x] 제안 멀티채널 CNN-GRU 뒤에 브레이크 전환용 룰 게이트를 추가한다.
- [x] 룰 게이트 적용 전후의 `accuracy`, `macro F1`, `boundary F1`, `brake_critical_recall` 변화를 측정한다.
- [x] 결과 표와 그래프를 루트 및 실험 폴더에 저장한다.
- [x] 동일 validation split에서 exploratory tuning했다는 점을 문서에 명시한다.

## 룰 게이트 개요
- [x] 중앙 ROI
- [x] center closeness
- [x] looming score
- [x] occlusion score
- [x] motion stop 성격
- [x] boundary + follow/warn 확률 승격

## 주의사항
- [x] validation split 하나만 사용하므로 룰 튜닝 결과는 낙관적일 수 있다.
- [x] 룰은 제안모델을 대체하는 것이 아니라 brake class 승격 보조로만 사용한다.

## 실제 결과
- [x] 실험 폴더
  - `experiments/exp_017_hybrid_rule_gate_benchmark/runs/context_model_benchmark_hybrid_v2`
- [x] 순수 제안모델
  - `context_acc=0.818333`
  - `context_macro_f1=0.236823`
  - `boundary_f1=0.050633`
  - `brake_critical_recall=0.111111`
- [x] 하이브리드 룰 게이트
  - `context_acc=0.800000`
  - `context_macro_f1=0.229760`
  - `boundary_f1=0.050633`
  - `brake_critical_recall=1.000000`

## 해석
- [x] 룰 게이트는 brake-critical recall을 크게 올린다.
- [x] 대신 전체 accuracy와 macro F1은 소폭 하락한다.
- [x] 즉 현재 하이브리드 방식은 브레이크 민감도 우선 설정으로 보는 것이 맞다.

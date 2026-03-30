# TASK_052_final_rule_reason_and_stopcar_threshold_tune.md

## 목표
- [x] `run_final_model.py`에 현재 프레임의 rule trigger reason 표시 추가
- [x] `final_hybrid_model_ui.py`의 하이브리드 최종 의사결정 로직 점검
- [x] `stopcar`에서 과한 `brake_warning` 구간 분석
- [x] `brake_warning` 과탐을 줄이도록 threshold 및 우선순위 조정
- [x] 관련 문서 갱신
- [x] 동작 검증

## 메모
- 현재 문제는 단순 threshold보다 `warn_rule`가 강한 `hard_brake_risk`를 `brake_warning`로 덮는 경우가 포함됨.
- rule trigger reason은 feature 기반 트리거와 signal 기반 트리거를 분리해서 표시한다.


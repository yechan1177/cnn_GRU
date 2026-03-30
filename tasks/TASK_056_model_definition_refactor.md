# TASK_056 모델 정의 재정렬 및 동일 데이터셋 재학습

## 목표
- [x] 1번 모델을 raw YOLO detection 기반 순간 규칙형으로 분리
- [x] 2번 모델을 동일 데이터셋 기반 CNN-GRU 단독 모델로 재정의
- [x] 3번 모델을 2번 모델 + 1번 규칙 보정형으로 재정의
- [x] 2번/3번 모델을 동일 데이터셋, 동일 학습 파이프라인으로 재학습
- [x] 세 모델 비교 실행기와 이벤트 저장 기준을 새 정의에 맞게 수정
- [x] 논문용 지표와 실제 문맥 일치 확률 수식 정리

## 작업 메모
- 1번 모델은 feature vector를 판단 입력으로 사용하지 않는다.
- 2번/3번 모델은 동일한 순간 feature sequence dataset을 사용한다.
- 3번 모델의 규칙 보정은 1번 모델의 raw detection rule을 그대로 사용한다.
- 실제 문맥 일치 확률은 사용자 수동 판정 기반 수식만 제공한다.

## 결과 요약
- 1번 모델은 `src/vcp/components/event_rules.py`의 raw detection 기반 순간 규칙만 사용하도록 정리했다.
- 순간 feature는 `src/vcp/components/spatial.py`에서 현재 프레임 기준 16차원으로 재구성했다.
- 공통 학습 데이터셋은 `data/processed/dataset_rule_context_instant_feat16_v2_20260327_162111`으로 고정했다.
- 2번/3번 temporal 체크포인트는 동일 학습 결과를 사용한다.
  - `models/checkpoints/temporal_shared_best.pt`
  - `models/checkpoints/temporal_final_best.pt`
- 세 모델 비교 산출물은 `artifacts/comparisons/<video>_three_model_compare/`에 저장된다.

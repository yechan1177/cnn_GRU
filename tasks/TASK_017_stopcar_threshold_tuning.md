# TASK_017: stopcar sudden_stop 임계값 튜닝

## 목표
- `stopcar.mp4` 기준으로 sudden_stop 점수 임계값을 정량적으로 튜닝한다.
- 트리거 중복을 줄이기 위해 최소 트리거 간격(min trigger gap) 조건을 함께 적용한다.
- 튜닝 결과를 파일로 저장하고 재실행 검증한다.

## 체크리스트
- [x] 태스크 문서 생성
- [x] 임계값 스윕 스크립트 구현
- [x] 추천 임계값/트리거 프레임 산출
- [x] 튜닝값으로 재실행 검증
- [x] README/TODO 반영

## 상태
- 완료

## 산출물(예정)
- `testing_work/tune_stopcar_threshold.py`
- `testing_work/outputs/<run_id>/tuning/threshold_sweep.csv`
- `testing_work/outputs/<run_id>/tuning/recommendation.json`

## 실제 결과
- 기준 run: `testing_work/outputs/stopcar_context_fusion_20260318_212428`
- 튜닝 결과: `threshold=0.51`, `min_trigger_gap=150`, `trigger_frames=352`
- 재검증 run: `testing_work/outputs/stopcar_context_fusion_20260318_212523`
  - `hard_brake_trigger_count=1`
  - `hard_brake_frame_count=18`

## 진행 메모
- 2026-03-18: 작업 시작
- 2026-03-18: threshold/gap 스윕 및 추천값 산출 완료
- 2026-03-18: 추천값 재실행 검증 완료

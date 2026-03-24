# TASK_015: stopcar 급정지 맥락 + 임의 가속도 센서 융합 테스트

## 목표
- PC 환경에서 `data/raw/videos/stopcar.mp4`를 입력으로 맥락 추론을 수행한다.
- 급정지(`hard_brake`) 맥락 신호가 발생하면 임의 가속도 센서 값이 점진적으로 감소하는 형태를 생성한다.
- 프레임 맥락과 센서 값을 시간축으로 정렬한 테스트 산출물을 `testing_work/`에 저장한다.

## 체크리스트
- [x] 태스크 문서 생성
- [x] `testing_work/` 폴더 생성
- [x] 실행 스크립트 구현
- [x] 정렬 출력 포맷(frame + sensor) 저장
- [x] stopcar.mp4 실제 실행 및 결과 확인
- [x] 문서 반영(TODO/README/가이드)

## 상태
- 완료

## 산출물(예정)
- `testing_work/run_stopcar_context_fusion.py`
- `testing_work/README.md`
- `testing_work/outputs/<run_id>/summary.json`
- `testing_work/outputs/<run_id>/frame_context.jsonl`
- `testing_work/outputs/<run_id>/synthetic_accel.csv`
- `testing_work/outputs/<run_id>/aligned_context_sensor.jsonl`

## 실제 결과(run)
- `testing_work/outputs/stopcar_context_fusion_20260318_204039/summary.json`
- `testing_work/outputs/stopcar_context_fusion_20260318_204039/frame_context.jsonl`
- `testing_work/outputs/stopcar_context_fusion_20260318_204039/synthetic_accel.csv`
- `testing_work/outputs/stopcar_context_fusion_20260318_204039/aligned_context_sensor.jsonl`

## 진행 메모
- 2026-03-18: 작업 시작
- 2026-03-18: stopcar.mp4 600프레임 실행 완료 (hard_brake trigger 5회, hard_brake frame 50프레임)

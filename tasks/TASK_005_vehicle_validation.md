# TASK_005: 차량 데이터셋 기반 구조 효용성 검증

## 목표
- 차량 내부에서 외부를 촬영한 데이터셋으로, 제안한 파이프라인 구조가 실제 데이터에서도 동작하는지 검증한다.
- 차량 비전 시스템 자체 성능 최적화가 아니라, 구조적 유효성(파이프라인 연결/정렬/정제/내보내기)을 확인하는 것을 1차 목표로 둔다.

## 체크리스트
- [x] 태스크 문서 생성
- [x] YOLO 데이터 구조 파악 및 입력 경로 확정
- [x] YOLO 라벨 -> run 포맷 변환 도구 구현
- [x] 변환 run 생성 및 통계 확인
- [x] 학습용 시퀀스 변환 실행
- [x] 실험 리포트/로그 문서화
- [x] 완료 상태 반영

## 산출물
- `src/vcp/tools/build_run_from_yolo_labels.py`
- `outputs/runs/<run_id>/...`
- `outputs/exports/<run_id>/...`
- `data/processed/dataset_<run_id>/...`
- `experiments/exp_003_vehicle_validation/...`

## 진행 메모
- 2026-03-18: 작업 시작
- 2026-03-18: `vehicle_validation_yolo6cls_20260318_150443` run 생성 및 processed dataset 변환 완료

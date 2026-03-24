# TASK_006: YOLO-nano 학습 및 파이프라인 연동 준비

## 목표
- `dataset/yolov8_6cls` 기준으로 YOLO-nano 학습을 수행한다.
- 학습 결과(체크포인트/로그/설정)를 실험 폴더에 구조화 저장한다.
- 이후 spatial encoder 연동 시 사용할 입력/출력 인터페이스를 문서화한다.

## 체크리스트
- [x] 태스크 문서 생성
- [x] 학습 환경 점검(ultralytics/torch)
- [x] data.yaml 경로 정합성 수정
- [x] YOLO-nano baseline 학습 실행
- [x] 결과 파일 정리(best.pt/metrics)
- [x] 연동 가이드 문서 업데이트
- [x] YOLO-nano 재학습 실행 (epochs=20, fraction=0.1)
- [x] 재학습 결과 지표/문서 반영
- [x] 완료 상태 반영

## 산출물
- `experiments/exp_004_yolo_nano_training/runs/yolov8n_e3_frac002_gpu/`
- `experiments/exp_004_yolo_nano_training/runs/yolov8n_e20_frac01_gpu/`
- `docs/09_YOLO학습_연동가이드.md`

## 진행 메모
- 2026-03-18: CUDA 11.8 + RTX 3080 Ti 환경에서 baseline(`e3,fraction=0.02`) 완료
- 2026-03-18: 재학습(`e20,fraction=0.1`) 완료, mAP50-95 0.1616 -> 0.2904 개선

## 상태
- 완료

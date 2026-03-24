# TASK_002: Spatial Encoder 고도화

## 목표
- mock spatial encoder에서 경량 실모델(YOLO-nano)로 전환 가능한 경로를 구현한다.
- 학습/실험(RTX 3080 Ti)과 배포/추론(Orin Nano)에서 동일 인터페이스로 동작하도록 모듈화한다.

## 체크리스트
- [x] 백본 후보 선정 및 추론 지연/메모리 예측
- [x] 모델 로더 인터페이스 구현 (`build_spatial_encoder`)
- [x] YOLO 가중치 로딩/추론/벡터화 구현 (`YOLOSpatialEncoder`)
- [x] feature packing 연계 검증
- [x] image folder 입력 스트림 구현 (`ImageFolderCameraStream`)
- [x] Jetson 배포 경량화 전략 문서 반영

## 산출물
- `src/vcp/components/spatial.py`
- `src/vcp/components/camera.py`
- `configs/runtime_rtx3080ti_yolo_gru.yaml`
- `outputs/runs/rtx3080ti_yolo_gru_runtime_20260318_164144/`

## 상태
- 완료

## 진행 메모
- 2026-03-18: `yolov8n_e20_frac01_gpu/weights/best.pt` 연동 가능한 runtime 경로 구현
- 2026-03-18: image_folder 입력 + YOLO spatial 연동 실행 검증(120 frame)

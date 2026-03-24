# TASK_032 YOLO 3클래스 학습 파이프라인 고도화 및 실행

## 목적
- [x] `person / vehicle / bike` 3클래스 데이터셋 기준 YOLO 학습 파이프라인을 정리한다.
- [x] `batch=16`, `epochs=150`, `optimizer=Adam`, `SiLU` 기반 학습 설정을 반영한다.
- [x] 증강 기반 1단계 학습 후 마지막 `N` epoch는 증강 없이 미세조정하는 2단계 구조를 구현한다.
- [x] 과적합 방지를 위해 early stopping(`patience`)을 반영한다.
- [x] `YOLOv8n`과 `YOLOv8s`를 Jetson Orin Nano 8GB 기준으로 비교하고, 적절한 모델을 선택한다.
- [x] 실제 학습 실행 결과를 실험 로그와 문서에 최종 반영한다.

## 작업 범위
- [x] 기존 YOLO 학습 스크립트 확장
- [x] 2단계 학습(stage1 augmented + stage2 no-aug) 구현
- [x] 모델 선택 근거 문서화
- [x] 장시간 본학습 완료 및 결과 요약
- [x] README/docs/experiment log 갱신

## 모델 선택 근거
- [x] `yolov8n.pt`: `129 layers`, `3,157,200 params`, `8.9 GFLOPs`
- [x] `yolov8s.pt`: `129 layers`, `11,166,560 params`, `28.8 GFLOPs`
- [x] Jetson Orin Nano 8GB 배포 제약을 고려해 기본 학습 모델은 `yolov8n.pt`로 선택
- [x] YOLOv8 기본 활성함수는 `SiLU`이므로 별도 커스텀 활성함수 교체는 수행하지 않음

## 현재 실행 설정
- [x] stage1: `epochs=130`, 증강 활성화
- [x] stage2: `final_epochs=20`, 증강 비활성화 미세조정
- [x] `optimizer=Adam`, `batch=16`, `patience=20`, `imgsz=640`
- [x] 데이터셋: `configs/datasets/yolo3cls_merged.yaml`
- [x] 시작 가중치: `models/pretrained/yolov8n.pt`

## 현재 실행 상태
- [x] 본학습 프로세스 시작
- [x] 본학습 종료 확인
- [x] 최종 best 체크포인트 경로 반영
- [x] stage2 no-aug 미세조정 완료 확인
- [x] 과적합 여부 및 조기 종료 여부 정리

## 중간 실행 메모
- 실행 run 이름: `yolov8n_3cls_from_pretrained`
- 로그 경로: `C:/yolstm/experiments/exp_011_yolo3cls_training/runs/yolov8n_3cls_from_pretrained_launcher/stdout.log`
- stage1 산출 경로: `C:/yolstm/experiments/exp_011_yolo3cls_training/runs/yolov8n_3cls_from_pretrained_stage1_aug`
- 1 epoch 검증 지표(중간값)
  - `precision(B)=0.56632`
  - `recall(B)=0.43463`
  - `mAP50(B)=0.46320`
  - `mAP50-95(B)=0.22045`

## 최종 결과 요약
- stage1 augmented 130 epoch 완료
  - `precision(B)=0.82020`
  - `recall(B)=0.65968`
  - `mAP50(B)=0.74085`
  - `mAP50-95(B)=0.42852`
  - best checkpoint: `C:/yolstm/experiments/exp_011_yolo3cls_training/runs/yolov8n_3cls_from_pretrained_stage1_aug/weights/best.pt`
- stage2 no-aug 미세조정
  - 요청 설정: `20 epoch`
  - 실제 종료: `19 epoch`
  - 조기 종료 사유: 최근 `10 epoch` 동안 개선 없음
  - best epoch: `9`
  - `precision(B)=0.82538`
  - `recall(B)=0.66244`
  - `mAP50(B)=0.73491`
  - `mAP50-95(B)=0.43404`
  - final best checkpoint: `C:/yolstm/experiments/exp_011_yolo3cls_training/runs/yolov8n_3cls_from_pretrained_stage2_noaug/weights/best.pt`

## 해석
- [x] `mAP50-95` 기준으로 stage2가 stage1보다 `+0.00552` 향상
- [x] `mAP50`는 소폭 감소했지만 (`0.74085 -> 0.73491`) 일반화 지표 `mAP50-95`는 개선
- [x] 심한 과적합으로 보기보다, no-aug 미세조정 구간에서 `epoch 9` 부근 이후 성능이 정체되어 조기 종료된 것으로 해석

## 주의사항
- [ ] 검증되지 않은 성능 수치를 문서에 단정하지 않는다.
- [ ] Jetson Orin Nano 배포를 고려한 과도한 모델 확장은 피한다.
- [ ] 실시간 학습 로그 중간 산출물과 설정을 반드시 저장한다.

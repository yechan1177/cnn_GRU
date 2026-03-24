# YOLO 3클래스 데이터셋 정리 가이드

## 목적
`dataset/` 하위의 YOLOv8 포맷 데이터셋을 아래 3개 클래스로 통일한다.

- `person`
- `vehicle`
- `bike`

## 매핑 규칙
### person
- `person`
- `pedestrian`

### vehicle
- `car`
- `truck`
- `bus`
- `cng`
- `other-vehicle`
- `rickshaw`

### bike
- `bicycle`
- `motorcycle`
- `bike`
- `biker`

### 제거 대상
- 신호등, 표지판 등 위 3개에 포함되지 않는 클래스

## 입력 데이터셋
- `dataset/P2_Dhaka_Dataset.v29i.yolov8`
- `dataset/Self Driving Car.v3-fixed-small.yolov8`

## 구조 차이
- `P2_Dhaka_Dataset.v29i.yolov8`
  - `train/valid/test` split이 이미 존재
- `Self Driving Car.v3-fixed-small.yolov8`
  - `export/images`, `export/labels` 평면 구조
  - 재매핑 시 `train/valid/test`로 재분할 필요

## 생성 명령
```bash
vcp-remap-yolo-3cls --dataset-root dataset --output-dir dataset/yolov8_3cls_merged
```

기본값:
- flat export split 비율: `train 0.8 / valid 0.1 / test 0.1`
- seed: `42`
- 링크 방식: `hardlink` 우선, 실패 시 `copy`

## 출력
- `dataset/yolov8_3cls_merged/data.yaml`
- `dataset/yolov8_3cls_merged/class_mapping.json`
- `dataset/yolov8_3cls_merged/remap_manifest.json`

## 학습 설정 파일
- `configs/datasets/yolo3cls_merged.yaml`

예시 학습:
```bash
vcp-train-yolo --data configs/datasets/yolo3cls_merged.yaml --model models/pretrained/yolov8n.pt --epochs 150 --final-epochs 20 --imgsz 640 --batch 16 --device 0 --optimizer Adam --patience 20 --project C:/yolstm/experiments/exp_011_yolo3cls_training/runs --name yolov8n_3cls_from_pretrained
```

## 모델 선택 근거
- `YOLOv8n`
  - 약 `3.16M` params
  - 약 `8.9 GFLOPs`
  - Jetson Orin Nano 8GB 배포 기준 권장
- `YOLOv8s`
  - 약 `11.17M` params
  - 약 `28.8 GFLOPs`
  - 정확도는 유리할 수 있으나 엣지 지연/메모리 부담이 큼

현재 프로젝트는 `Jetson Orin Nano 8GB`를 실제 배포 타깃으로 두므로, 우선 `YOLOv8n`을 채택한다.

## 학습 전략
- Stage 1:
  - Adam 옵티마이저
  - 증강 활성화
  - `patience` 기반 early stopping
- Stage 2:
  - Stage 1 best checkpoint에서 시작
  - 마지막 `N` epoch는 증강 비활성
  - 실제 추론 분포에 가까운 미세조정

참고:
- YOLOv8 기본 Conv 블록 활성함수는 `SiLU`를 사용한다.

## 주의
- 원본 데이터셋은 수정하지 않는다.
- 재매핑 결과는 별도 폴더에 생성한다.
- `biker`는 `이륜차 + 탑승자` 성격의 복합 객체지만, 현재 연구 목적상 `bike`로 통합한다.

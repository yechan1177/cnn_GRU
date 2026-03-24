# 09. YOLO 학습 및 연동 가이드

## 목적
RTX 3080 Ti + CUDA 11.8 환경에서 YOLO-nano를 학습하고,
Spatial Encoder 입력으로 연동한다.

## 환경
- venv: `.venv`
- PyTorch: CUDA 11.8 빌드
- Ultralytics: `ultralytics>=8.4.0`

## 데이터셋
- data yaml: `configs/datasets/yolo6cls_local.yaml`
- root: `dataset/yolov8_6cls_local`

## 학습 실행
baseline:
```bash
vcp-train-yolo --data configs/datasets/yolo6cls_local.yaml --epochs 3 --imgsz 640 --batch 16 --fraction 0.02 --device 0 --project C:/yolstm/experiments/exp_004_yolo_nano_training/runs --name yolov8n_e3_frac002_gpu
```

재학습:
```bash
vcp-train-yolo --data configs/datasets/yolo6cls_local.yaml --epochs 20 --imgsz 640 --batch 16 --fraction 0.1 --device 0 --project C:/yolstm/experiments/exp_004_yolo_nano_training/runs --name yolov8n_e20_frac01_gpu
```

## 결과 요약
- e3/f0.02: `precision 0.4229`, `recall 0.2911`, `mAP50-95 0.1616`
- e20/f0.1: `precision 0.5804`, `recall 0.4579`, `mAP50-95 0.2904`

## 파이프라인 연동
- Spatial: `src/vcp/components/spatial.py` 의 `YOLOSpatialEncoder`
- Runtime config 예시: `configs/runtime_rtx3080ti_yolo_gru.yaml`
- 출력 벡터는 기존 `feature packing -> temporal` 경로와 동일 인터페이스로 연결된다.

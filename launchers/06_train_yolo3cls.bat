@echo off
cd /d C:\yolstm
call .\.venv\Scripts\activate.bat
python -m vcp.tools.train_yolo_nano --data configs/datasets/yolo3cls_merged.yaml --model models/pretrained/yolov8n.pt --epochs 150 --final-epochs 20 --imgsz 640 --batch 16 --device 0 --optimizer Adam --patience 20 --project C:/yolstm/experiments/exp_011_yolo3cls_training/runs --name yolov8n_3cls_from_pretrained

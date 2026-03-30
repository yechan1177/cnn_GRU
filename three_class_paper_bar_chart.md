# 주요 3개 상황 분포 비교

- 기준 파일: `C:\yolstm\artifacts\comparisons\people_braking_three_model_compare\summary.json`

| 상황 | YOLO+rule | CNN-GRU only | CNN-GRU+rule |
|---|---:|---:|---:|
| 일반 주행 (normal_drive) | 29 | 37 | 270 |
| 차량 추종 (vehicle_follow) | 368 | 102 | 123 |
| 브레이크 경고 (brake_warning) | 25 | 279 | 27 |

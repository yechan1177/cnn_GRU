# models

## 저장소에 포함된 체크포인트 (`checkpoints/`)
| 파일 | 내용 | 비고 |
|---|---|---|
| `yolo3cls_best.pt` | YOLOv8n 3클래스(person/vehicle/bike) | 약 301만 파라미터 |
| `temporal_final_best.pt` | 2026-03 최종 멀티채널 CNN-GRU(exp_020, 증강 데이터) | v1 특징, 채널 그룹 = 인덱스 균등 분할 `[0-4],[5-8],[9-12],[13-15]`, 약 6.5만 파라미터 |
| `literature_temporal_best.pt` | 2026-03 문헌형 단일채널 CNN-GRU 비교군 | `LiteratureCNNGRUNet` 구조 |

### 주의
- 2026-03 비교표(docs/21)는 exp_015 체크포인트로 계산되었고, 위 `temporal_final_best.pt`는 exp_020이다.
- 두 temporal 체크포인트는 v1 특징 전용이다. v2 특징 모델은 자동 실험 스위트에서 학습된다(`docs/27`).

## 로컬 전용(저장소 미포함)
- `pretrained/`: 외부 사전학습 가중치(`yolov8n.pt` 등)
- 실험 학습 결과: `experiments/<exp>/runs/`, `experiments/exp_100_paper_suite/checkpoints/`

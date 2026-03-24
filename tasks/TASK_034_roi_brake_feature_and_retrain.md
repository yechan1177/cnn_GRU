# TASK_034 ROI/브레이크 전환 feature 추가 및 맥락 재학습

## 목적
- [x] 중앙 ROI 기반 위험도 feature를 추가한다.
- [x] 짧은 프레임 사이 bbox 급팽창(looming) 신호를 feature에 반영한다.
- [x] 움직임 둔화/정지(stop-motion) 신호를 feature에 반영한다.
- [x] `stopcar` annotation을 `brake_warning -> hard_brake_risk` 전환 중심으로 더 촘촘히 재작성한다.
- [x] 새 feature 기반 dataset과 GRU 재학습 결과를 정리한다.

## 작업 범위
- [x] spatial encoder feature 설계 갱신
- [x] feature key 저장 구조 반영
- [x] 16차원 runtime/학습 경로 정리
- [x] stopcar annotation 세분화
- [x] 새 run/dataset/temporal 재학습
- [x] 문서/실험 로그 반영

## 설계 메모
- 중앙 ROI는 화면 상하를 일부 제외하고 중앙 폭 위주로 정의한다.
- `brake_warning`: 중앙 ROI에서 bbox가 짧은 프레임 사이 빠르게 커지는 구간
- `hard_brake_risk`: bbox가 더 급격히 커지거나 motion이 둔화/정지하는 구간
- negative mining은 이번 단계에서 수행하지 않고 detector threshold와 feature 설계로 대응한다.

## 구현 결과
- 새 16차원 feature
  - `det_norm`
  - `mean_conf`
  - `max_conf`
  - `mean_area`
  - `area_var`
  - `vehicle_ratio`
  - `person_ratio`
  - `bike_ratio`
  - `vehicle_conf`
  - `person_conf`
  - `bike_conf`
  - `roi_risk`
  - `motion_delta`
  - `center_closeness`
  - `looming_score`
  - `occlusion_score` (`stop-motion` 성격)
- 새 detector feature run
  - `stopcar_yolo3cls_feat16_t045_demo_20260324_105046`
  - `race_yolo3cls_feat16_t045_demo_20260324_105046`
- 새 brake transition dataset
  - `data/processed/dataset_manual_context_yolo3cls_feat16_t045_brake_20260324_105216`
- 새 temporal 학습
  - run: `exp_014_temporal_brake_transition_feat16`
  - best epoch: `2`
  - `val_context_acc=0.797521`
  - `val_boundary_f1=0.0`

## 검증 결과
- `stopcar` 재추론 run: `stopcar_brake_transition_t045_demo_20260324_105245`
- 기대: `333~342 -> brake_warning`, `343~350 -> hard_brake_risk`
- 실제: 주로 `normal_drive`, 일부 `front_vehicle_follow`
- 결론: feature와 annotation 세분화는 반영됐지만 현재 표본 수와 detector 오분류 때문에 목표 라벨 분리는 아직 실패

## 한계
- 핵심 stopcar 프레임 일부에서 detector가 여전히 전방 객체를 `person`으로 오인한다.
- `brake_warning` 표본 `10`, `hard_brake_risk` 표본 `8`로 매우 적다.
- validation split에는 rare brake class가 포함되지 못했다.

# 브레이크 positive feature 합성 증강

## 목적
- 희소한 `brake_warning`, `hard_brake_risk` 표본을 image-level이 아니라 feature-level에서 증강한다.
- detector를 다시 학습하지 않고 temporal GRU의 브레이크 구간 일반화를 보강한다.

## 방법
- 대상: `train.jsonl`의 `brake_warning`, `hard_brake_risk` 샘플만 사용
- 방식:
  - 작은 전역 noise 추가
  - `vehicle_ratio` 증가, `person_ratio` 감소
  - `roi_risk`, `center_closeness`, `looming_score`, `occlusion_score`를 라벨에 맞게 강화
  - `hard_brake_risk`는 마지막 프레임 쪽에서 `looming`과 `occlusion`을 더 강하게 증폭
- 제한:
  - 모든 feature는 `0~1`로 클리핑
  - val split은 증강하지 않음

## 증강 데이터셋
- 원본: `data/processed/dataset_manual_context_braking_expanded_feat16_t045_20260324_115253`
- 증강본: `data/processed/dataset_manual_context_braking_expanded_feat16_t045_augmented_20260324`

## 표본 변화
- 원본 train:
  - `brake_warning=24`
  - `hard_brake_risk=28`
- 증강 후 train:
  - `brake_warning=168`
  - `hard_brake_risk=308`

## 학습 결과
- 실험: `exp_020_temporal_brake_positive_augmented`
- best epoch: `13`
- `val_context_acc=0.831667`
- `val_boundary_f1=0.089888`

## people_braking 결과
- 기준 checkpoint(`exp_015`) final_ui_hybrid:
  - `context_match_prob=0.538462`
  - `brake_critical_recall=0.230769`
- 증강 checkpoint(`exp_020`) final_ui_hybrid:
  - `context_match_prob=0.666667`
  - `brake_critical_recall=0.230769`

## 해석
- 합성 증강은 이번 `people_braking` 영상에서 `브레이크 recall`을 더 끌어올리진 못했다.
- 대신 `normal_drive`, `front_vehicle_follow`, `brake_warning` 사이의 전체 문맥 정합도를 올리는 효과는 확인됐다.
- 따라서 현재 증강 방식은 `브레이크 민감도 강화`보다 `문맥 해석 안정화` 쪽 효과가 더 크다.

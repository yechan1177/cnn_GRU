# TASK_045 브레이크 positive feature 합성 증강

## 목적
- [x] `brake_warning`, `hard_brake_risk` 희소 표본을 feature 공간에서 합성 증강한다.
- [x] train split만 증강한 새 dataset을 생성한다.
- [x] 증강 dataset으로 GRU를 재학습한다.
- [x] `people_braking.mp4` 기준으로 문맥 일치 확률과 brake recall 개선 여부를 확인한다.

## 구현 계획
- [x] 기존 processed dataset의 train.jsonl을 읽는다.
- [x] brake positive 샘플만 대상으로 ROI/looming/center/occlusion 관련 합성 변형을 만든다.
- [x] 증강 dataset을 별도 폴더로 저장한다.
- [x] 새 temporal checkpoint를 학습한다.
- [x] people_braking 3모델 비교에 증강 제안모델 결과를 추가 측정한다.

## 주의사항
- [x] val split은 증강하지 않는다.
- [x] 실제 수집 데이터가 아님을 문서에 명시한다.
- [x] 과도한 값 왜곡을 막기 위해 feature 범위를 `0~1`로 클리핑한다.

## 결과 요약
- [x] 증강 dataset 생성
  - `dataset_manual_context_braking_expanded_feat16_t045_augmented_20260324`
- [x] 증강 학습 결과
  - `val_context_acc=0.831667`
  - `val_boundary_f1=0.089888`
- [x] people_braking 기준 비교
  - 기존 final_ui_hybrid `context_match_prob=0.538462`
  - 증강 final_ui_hybrid `context_match_prob=0.666667`
  - `brake_critical_recall`은 둘 다 `0.230769`

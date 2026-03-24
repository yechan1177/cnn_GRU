# TASK_035 추가 급제동 영상 표본 확장 및 GRU 재학습

## 목적
- [x] 추가 급제동 영상 4개를 detector feature run으로 변환한다.
- [x] looming/occlusion/roi peak 기반으로 brake transition draft annotation을 자동 생성한다.
- [x] 확장된 표본으로 context dataset을 재구축한다.
- [x] 확장된 dataset으로 GRU를 재학습한다.
- [x] stopcar 기준으로 재검증한다.

## 대상 영상
- [x] people_braking.mp4
- [x] rainy_stopcar.mp4
- [x] stopcar1.mp4
- [x] stopcar2.mp4

## 주의사항
- [x] 자동 annotation은 draft로만 저장한다.
- [x] 표본 확장이 곧 GT 확보를 의미하지는 않는다.
- [x] detector 오분류가 있더라도 looming/roi 신호 기반으로 brake transition을 우선 본다.

## 구현 결과
- [x] 추가 영상 detector feature run 생성
  - `people_braking_yolo3cls_feat16_t045_demo_20260324_115132`
  - `rainy_stopcar_yolo3cls_feat16_t045_demo_20260324_115142`
  - `stopcar1_yolo3cls_feat16_t045_demo_20260324_115153`
  - `stopcar2_yolo3cls_feat16_t045_demo_20260324_115205`
- [x] peak 기반 draft annotation 생성
  - `people_braking_context_segments.jsonl`
  - `rainy_stopcar_context_segments.jsonl`
  - `stopcar1_context_segments.jsonl`
  - `stopcar2_context_segments.jsonl`
- [x] 확장 dataset 생성
  - `data/processed/dataset_manual_context_braking_expanded_feat16_t045_20260324_115253`
  - 총 sample `4,637`, validation `600`
- [x] 확장 temporal 재학습
  - 실험: `exp_015_temporal_braking_expanded_feat16`
  - best epoch: `16`
  - `val_context_acc=0.818333`
  - `val_boundary_f1=0.050633`
  - `val_boundary_recall=1.000000`
- [x] stopcar 재검증
  - run: `stopcar_braking_expanded_t045_demo_20260324_115331`
  - 전체 분포: `normal_drive 531`, `front_vehicle_follow 57`, `brake_warning 3`, `post_brake_recovery 9`
  - `hard_brake_risk`는 출력되지 않음

## 해석
- [x] 추가 급제동 영상 표본 확장으로 `brake_warning`는 일부 프레임에서 처음 출력되기 시작했다.
- [x] 그러나 `hard_brake_risk`는 아직 안정적으로 분리되지 않았다.
- [x] 현재 병목은 모델 구조보다도 draft annotation 품질, rare class 표본 부족, detector 오분류 영향에 더 가깝다.

## 다음 단계
- [ ] 자동 draft annotation을 사람 검수 annotation으로 승격한다.
- [ ] `hard_brake_risk` 구간을 더 길고 일관되게 다시 라벨링한다.
- [ ] detector 기반 feature 외에 rule gate를 함께 둬서 `brake_warning -> hard_brake_risk` 전환을 보조한다.

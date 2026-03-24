# TASK_033 YOLO 3클래스 추론 검증 및 맥락 모델 재학습

## 목적
- [x] 새 YOLO 3클래스 `best.pt`를 실제 영상(`stopcar`, `race`)에 연결해 추론한다.
- [x] 검출 품질을 프레임/클래스 통계 기준으로 점검한다.
- [x] 새 detector 결과를 기준으로 수동 맥락 dataset을 재생성한다.
- [x] 새 dataset으로 temporal 맥락 모델을 재학습한다.
- [x] 결과를 실험 로그와 요약 문서에 정리한다.

## 작업 범위
- [x] runtime config 정리
- [x] frame_records feature 저장 구조 확장
- [x] stopcar/race run 재생성
- [x] annotation run_id 갱신
- [x] context dataset 재생성
- [x] temporal 재학습

## 주의사항
- [ ] 새 detector와 old temporal checkpoint 의미를 혼동하지 않는다.
- [ ] 검출 품질 평가는 실제 GT가 아니라 run 통계/시각 확인 기반의 예비 점검임을 명시한다.
- [ ] 문서 갱신 없이 코드만 수정하지 않는다.

## 결과 요약
- threshold 스윕 결과 `conf_threshold=0.45`를 선택
  - `stopcar.mp4`
    - `0.25`: person `38`, vehicle `932`
    - `0.45`: person `18`, vehicle `709`
  - `race.mp4`
    - `0.25`: person `114`, vehicle `11337`
    - `0.45`: person `72`, vehicle `8890`
- 새 feature 저장 구조
  - `frame_records.derived_feature.feature_vector`에 64차원 전체 spatial vector 저장
- 새 run
  - `stopcar_yolo3cls_t045_demo_20260324_104143`
  - `race_yolo3cls_t045_demo_20260324_104143`
- 새 dataset
  - `data/processed/dataset_manual_context_yolo3cls_t045_20260324_104245`
- 새 temporal 학습
  - run: `exp_013_temporal_manual_context_yolo3cls_t045`
  - best epoch: `3`
  - `val_context_acc=0.694561`
  - `val_boundary_f1=0.026316`
- 새 detector + 새 temporal로 `stopcar` 재검증
  - run: `stopcar_yolo3cls_mctx_t045_v3_demo_20260324_104339`
  - 맥락 분포: `normal_drive 451`, `front_vehicle_follow 149`
  - 급정지 구간(`335~360`)은 전부 `front_vehicle_follow`로 출력

## 해석
- threshold 조정만으로 `person` 오탐은 줄었지만 제거되지는 않았다.
- 새 feature 기반 GRU 재학습 후, 기존 `vehicle` 단일 응답보다는 `front_vehicle_follow`로 의미가 조금 정리되었다.
- 하지만 현재 draft annotation 수가 적고 `hard_brake_risk` 표본이 매우 적어, `stopcar` 급정지 구간을 `hard_brake_risk`로 분리하는 데는 아직 실패했다.

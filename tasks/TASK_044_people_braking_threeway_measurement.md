# TASK_044 people_braking 3모델 실측 비교

## 목적
- [x] `people_braking.mp4` 기준으로 3모델을 다시 실측 비교한다.
- [x] 비교군은 `YOLO+rule`, `문헌형 CNN+시계열`, `최종 UI 하이브리드`로 고정한다.
- [x] `실제 문맥 일치 확률`, `브레이크 민감도`, `추론 시간`, `파라미터 수`를 다시 계산한다.
- [x] `객체인식 정확도`는 bbox GT 부재를 명시하고 공통 detector의 validation `mAP50`을 사용한다.
- [x] 루트에 발표/논문용 표와 그래프를 새 파일로 저장한다.

## 구현 계획
- [x] `people_braking` annotation을 프레임 GT로 변환한다.
- [x] 3모델을 `people_braking.mp4`에 대해 프레임 단위로 다시 추론한다.
- [x] context accuracy / brake-critical recall을 직접 계산한다.
- [x] 같은 영상/같은 장치 기준 추론 시간을 측정한다.
- [x] 루트 산출물과 문서를 갱신한다.

## 주의사항
- [x] 기존 `stopcar` 비교 파일은 덮어쓰지 않는다.
- [x] bbox GT가 없으므로 detector mAP50은 공통 지표로만 사용한다.
- [x] `draft` annotation 기반 결과임을 문서에 명시한다.

## 실측 결과
- [x] `YOLO+rule`
  - `context_match_prob=0.414918`
  - `brake_critical_recall=0.230769`
  - `inference=7.6599 ms/frame`
- [x] `literature_cnn_temporal`
  - `context_match_prob=0.463869`
  - `brake_critical_recall=0.076923`
  - `inference=8.0349 ms/frame`
- [x] `final_ui_hybrid`
  - `context_match_prob=0.538462`
  - `brake_critical_recall=0.230769`
  - `inference=8.6778 ms/frame`

## 산출물
- [x] `people_braking_threeway_comparison.csv`
- [x] `people_braking_threeway_comparison.md`
- [x] `people_braking_threeway_comparison.png`
- [x] `people_braking_threeway_comparison_manifest.json`

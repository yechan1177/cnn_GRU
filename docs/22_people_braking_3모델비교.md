# people_braking 3모델 비교

## 비교 대상
- `YOLO+rule`
- `literature_cnn_temporal`
- `final_ui_hybrid`

## 비교 조건
- 입력 영상: `data/raw/videos/people_braking.mp4`
- 문맥 GT: `data/annotations/context/people_braking_context_segments.jsonl`
- detector: `exp_011`의 YOLO 3클래스 가중치 공통 사용
- detector threshold: `0.65`
- 주의: 이 영상에는 bbox GT가 없으므로 `객체인식 정확도`는 detector validation `mAP50` 공통값을 사용했다.
- 주의: 문맥 annotation 상태는 `draft`이다.

## 실제 결과

| 모델 | 실제 문맥 일치 확률 | 객체인식 정확도(mAP50) | 추론 시간(ms/frame) | 파라미터 수 | brake_critical_recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO+rule | 0.414918 | 0.734910 | 7.6599 | 3,011,433 | 0.230769 |
| literature_cnn_temporal | 0.463869 | 0.734910 | 8.0349 | 3,050,224 | 0.076923 |
| final_ui_hybrid | 0.538462 | 0.734910 | 8.6778 | 3,076,529 | 0.230769 |

## 해석
- `people_braking`에서는 `final_ui_hybrid`가 가장 높은 `실제 문맥 일치 확률`을 보였다.
- `brake_critical_recall`은 `YOLO+rule`과 `final_ui_hybrid`가 동일했고, 문헌형 baseline은 더 낮았다.
- 즉 이 영상에서는 하이브리드 모델의 장점이 `브레이크 민감도 추가 상승`보다 `전체 문맥 일치 확률 개선` 쪽으로 나타났다.
- 세 모델은 같은 detector를 공유하므로 `객체인식 정확도(mAP50)`는 동일하다.

## 루트 산출물
- `people_braking_threeway_comparison.csv`
- `people_braking_threeway_comparison.md`
- `people_braking_threeway_comparison.png`
- `people_braking_threeway_comparison_manifest.json`

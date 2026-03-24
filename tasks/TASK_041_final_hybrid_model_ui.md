# TASK_041 최종 하이브리드 모델 확인 UI

## 목적
- [x] 최종 제안모델과 하이브리드 룰 게이트 결과를 한 화면에서 확인하는 UI를 만든다.
- [x] 저장된 run을 다시 읽어서 순수 모델 예측과 최종 하이브리드 태그를 함께 표시한다.
- [x] 발표/논문용으로 바로 쓸 수 있는 스크린샷을 산출한다.
- [x] 실행 파일 위치와 사용법을 문서에 반영한다.

## 표시 항목
- [x] 현재 프레임
- [x] 순수 모델 top1 context
- [x] 하이브리드 최종 context
- [x] boundary score
- [x] brake_warning / hard_brake_risk 확률
- [x] 최근 프레임 추이 그래프

## 주의사항
- [x] run에 저장된 feature vector를 기준으로 재계산한다.
- [x] detector를 다시 돌리지 않는다.
- [x] 기본 하이브리드 파라미터는 benchmark manifest 기준으로 불러온다.

## 구현 결과
- [x] UI 스크립트 추가
  - `src/vcp/tools/final_hybrid_model_ui.py`
- [x] 실행 배치 추가
  - `launchers/08_final_hybrid_model_ui.bat`
- [x] 콘솔 진입점 추가
  - `vcp-final-model-ui`
- [x] 가이드 문서 추가
  - `docs/20_최종하이브리드모델UI.md`
- [x] 프리뷰 스크린샷 생성
  - `artifacts/screenshots/final_hybrid_model_ui_preview.jpg`

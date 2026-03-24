# TASK_016: stopcar 결과 확인용 UI 대시보드 구현

## 목표
- `testing_work/outputs`에 저장된 급정지 융합 결과를 사용자가 시각적으로 검증할 수 있는 UI를 제공한다.
- 프레임 탐색, 급정지 점수(sudden_stop), 합성 가속도(ax), 컨텍스트 태그를 한 화면에서 확인 가능하게 한다.

## 체크리스트
- [x] 태스크 문서 생성
- [x] UI 스크립트 구현
- [x] 프레임 탐색(트랙바/키보드) 지원
- [x] 점수/센서 시계열 패널 표시
- [x] 실행 검증
- [x] 문서 반영

## 상태
- 완료

## 산출물(예정)
- `testing_work/stopcar_review_ui.py`
- `testing_work/README.md` (UI 실행법 갱신)
- `testing_work/outputs/<run_id>/dashboard_preview.jpg` (검증 산출물)

## 실제 검증 산출물
- `testing_work/outputs/stopcar_context_fusion_20260318_204039/dashboard_preview.jpg`

## 진행 메모
- 2026-03-18: 작업 시작
- 2026-03-18: UI 구현 및 프리뷰 이미지 생성으로 동작 검증 완료

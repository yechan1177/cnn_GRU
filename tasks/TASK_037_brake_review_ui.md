# TASK_037 브레이크 구간 검수 UI 추가

## 목적
- [x] 현재 braking run을 직접 검수할 수 있는 새 UI를 구현한다.
- [x] 프레임 이동, 라벨 선택, 시작/끝 지정, boundary 지정 기능을 제공한다.
- [x] reviewed annotation을 jsonl로 저장한다.
- [x] 사용자 실행용 배치 파일을 추가한다.
- [x] 테스트와 문서를 갱신한다.

## 구현 범위
- [x] `src/vcp/tools/brake_review_ui.py`
- [x] `launchers/07_brake_review_ui.bat`
- [x] `tests/test_brake_review_ui.py`
- [x] `docs/17_브레이크리뷰UI.md`
- [x] `pyproject.toml` console script 반영

## 결과
- [x] 최신 braking run 자동 탐색
- [x] `1~6` 라벨 선택, `z/x` 구간 마킹, `b` boundary 토글, `c` 구간 추가, `w` 저장 지원
- [x] reviewed annotation 기본 저장 경로: `data/annotations/context/<video_stem>_reviewed_context_segments.jsonl`
- [x] `pytest -q` 통과

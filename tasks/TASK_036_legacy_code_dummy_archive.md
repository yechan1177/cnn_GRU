# TASK_036 legacy 코드 dummy 폴더 아카이브

## 목적
- [x] 현재 메인 학습/추론 경로와 무관한 legacy 코드를 식별한다.
- [x] legacy 코드를 `dummy/legacy_archive_20260324/`로 이동한다.
- [x] pyproject 진입점에서 legacy 스크립트 참조를 제거한다.
- [x] README/launcher 문서에서 legacy 경로 안내를 정리한다.
- [x] 현재 코어 파이프라인이 유지되는지 검증한다.

## 아카이브 대상
- [x] `testing_work/` 기반 legacy stopcar 테스트 코드
- [x] `src/vcp/tools/`의 예전 변환/합성/시나리오 스크립트 일부
- [x] 관련 launcher/test 코드

## 주의사항
- [x] 현재 사용하는 `run_video_demo.py`, `build_context_dataset.py`, `train_yolo_nano.py`, `train_temporal_gru.py`는 유지한다.
- [x] 실험 산출물과 데이터는 삭제하지 않는다.
- [x] 이동 후 문서와 진입점을 같이 정리한다.

## 결과
- [x] `dummy/legacy_archive_20260324/` 생성
- [x] legacy code/test/launcher 이동
- [x] `pyproject.toml`에서 legacy console script 제거
- [x] `README.md`, `launchers/README.md` 정리
- [x] `pytest -q` 통과 (`8 passed`)

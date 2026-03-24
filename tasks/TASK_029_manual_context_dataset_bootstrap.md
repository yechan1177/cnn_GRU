# TASK_029 수동 맥락 라벨 데이터셋 부트스트랩

## 목적
- [x] `YOLO 검출`과 `맥락 분류`를 분리 설명할 수 있는 수동 맥락 라벨 체계를 정의한다.
- [x] `stopcar.mp4`, `race.mp4`에 대한 초안 구간 라벨 파일을 생성한다.
- [x] 수동 라벨 구간을 `Temporal GRU` 학습용 시퀀스 데이터셋으로 변환하는 스크립트를 추가한다.
- [x] README 또는 설계 문서에 사용 방법과 주의사항을 반영한다.
- [x] 테스트 코드를 추가해 최소 변환 경로를 검증한다.

## 작업 범위
- [x] `data/annotations/context/` 폴더 및 라벨 스키마 추가
- [x] `stopcar`, `race` 초안 annotation JSONL 작성
- [x] `src/vcp/tools/build_context_dataset.py` 구현
- [x] `pyproject.toml` 엔트리포인트 추가
- [x] `docs/14_맥락라벨링_데이터셋가이드.md` 작성
- [x] `README.md` 갱신
- [x] 실제 초안 데이터셋 1회 생성
- [x] `pytest -q` 검증

## 산출물
- [x] 수동 맥락 라벨 스키마 파일
- [x] 영상별 초안 annotation 파일
- [x] 수동 맥락 학습용 processed dataset
- [x] 가이드 문서

## 주의사항
- [x] 현재 annotation은 `초안(draft)`이며, 검증된 GT로 표기하지 않는다.
- [x] `people` 같은 기존 약한 라벨 대신 `상황(context)` 중심 라벨을 사용한다.
- [x] 현재 run feature는 `derived_feature.feature_sample(8차원)` 기준이므로 표현력 한계를 문서에 명시한다.

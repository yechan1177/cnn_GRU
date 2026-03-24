# TASK_021: proposal_v2 기준 파트별 재구성

## 목표
- `proposal/source/proposal_v2.md`의 구조(입력~export 6단계)에 맞춰 코드/문서/연구 산출물 접근 구조를 재정렬한다.
- 기존 실행 경로를 최대한 유지하면서, 제안서/연구 작성 시 바로 참조 가능한 파트별 폴더를 제공한다.

## 체크리스트
- [x] 태스크 문서 생성
- [x] `src` 파트별 패키지(`vcp.parts`) 구성
- [x] 파이프라인 import 경로를 파트 패키지 기준으로 정렬
- [x] `research/parts` 폴더 구조 및 파트별 인덱스 문서 생성
- [x] `docs` 파트 매핑 문서 추가
- [x] `README` 갱신
- [x] 기본 테스트(스모크) 실행
- [x] TODO 반영

## 상태
- 완료

## 진행 메모
- 2026-03-18: 작업 시작
- 2026-03-18: `src/vcp/parts` 패키지 추가 및 파이프라인 연동
- 2026-03-18: `research/parts`, `experiments/by_part`, `artifacts/by_part` 인덱스 구조 추가
- 2026-03-18: `pytest -q` 7건 통과, `python -m vcp.main --config configs/default.yaml` 실행 확인

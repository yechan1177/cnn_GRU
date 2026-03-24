# TASK_001: 프로젝트 초기 구조/문서/베이스라인 파이프라인 구축

## 목표
- 요청된 폴더 구조를 생성한다.
- 한국어 문서(README/설계/실험 정리)를 작성한다.
- 학습 환경(RTX 3080 Ti)과 배포 환경(Jetson Orin Nano 8GB) 분리 전략을 config와 코드 구조에 반영한다.
- mock 기반 end-to-end 파이프라인과 테스트를 제공한다.

## 체크리스트
- [x] 태스크 문서 생성
- [x] 프로젝트 루트 구조 생성
- [x] Python 패키지 및 core interfaces 작성
- [x] baseline 파이프라인 구현
- [x] 설정 파일(default/train/deploy) 작성
- [x] 테스트 코드 작성
- [x] README 및 docs 동기화
- [x] experiments/artifacts 템플릿 작성
- [x] 검증 실행(대체 검증: `python -m compileall src tests`)
- [x] 완료 상태 갱신

## 산출물
- `README.md`
- `docs/00_프로젝트개요.md`
- `docs/01_시스템아키텍처.md`
- `docs/02_데이터스키마.md`
- `docs/03_모델설계.md`
- `docs/04_배포전략.md`
- `docs/05_실험방법.md`
- `src/...`
- `tests/...`
- `experiments/...`
- `artifacts/...`

## 진행 메모
- 2026-03-18: 작업 시작
- 2026-03-18: baseline 구조 구현 완료, `pytest`/`yaml` 미설치로 런타임 테스트는 보류
- 2026-03-18: temporal/curation 개선으로 이벤트 트리거 안정화, `.venv` 기준 `pytest` 2건 통과

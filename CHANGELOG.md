# Changelog

이 문서는 GitHub 배포 기준으로 의미 있는 저장소 변경만 기록한다.

## 2026-04-03
### Added
- GitHub만으로 현재 상태를 이어받을 수 있도록 `docs/30_프로젝트_종합인수인계.md` 추가
- Jetson 배포용 `docs/31_jetson_설치체크리스트.md` 추가
- Jetson 실행 명령 중심의 `docs/32_jetson_배포실행매뉴얼.md` 추가
- Jetson 성능 검증 표 양식 `docs/33_jetson_성능검증표양식.md` 및 `jetson_performance_validation_template.csv` 추가

### Changed
- README에 새 스레드/인수인계 시작 경로 추가
- 문서 인덱스에 종합 인수인계 문서 연결
- 논문 재현 브랜치 기준 현재 모델 정의와 산출물 위치를 종합 문서에 정리
- README와 배포 전략 문서에 Jetson 배포 경로 연결

## 2026-03-24
### Added
- GitHub 배포용 `.gitignore` 추가
- 루트 단일 실행 파일 `run_final_model.py` 정리
- 실행용 최소 체크포인트 `models/checkpoints/` 구성
- 하이브리드 룰 파라미터 `configs/hybrid_rule_params.json` 추가
- README 공식 문서형 개편
- README용 SVG 파이프라인 다이어그램 추가

### Changed
- 절대경로 기반 실행을 상대경로 기반으로 정리
- 패키지 의존성 정의를 실행 중심으로 정리
- 최종 UI 기본 체크포인트 경로를 저장소 기준으로 변경

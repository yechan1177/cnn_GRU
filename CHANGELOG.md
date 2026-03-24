# Changelog

이 문서는 GitHub 배포 기준으로 의미 있는 저장소 변경만 기록한다.

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

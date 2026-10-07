# Changelog

이 문서는 GitHub 배포 기준으로 의미 있는 저장소 변경만 기록한다.

## 2026-10-02
### Added
- `vcp.sim.env`(단계형 SimEnv, 기존 `simulate_episode` 비트 동일 회귀), `vcp.sim.render`(64×64 렌더러)
- `vcp.vla`: `instructions`(주행 스타일 지시문), `obs`, `pool`(지시문 데이터 풀), `closed_loop`(폐루프 평가), `policy`(VLA-lite)
- `vcp.vla.curation`: 선별법 10종(`select`, `selection_stats`)
- `vcp.experiments.vla_curation_suite`, `comma_curation`, `vla_report`(대응 부트스트랩·스피어만·표·그림)
- `scripts/run_vla_curation.sh`, `build_vla_paper.py`, `plot_vla_qualitative.py`, `render_svg.py`, `extract_small_frames.py`
- VLA 연계 논문 원고(`paper/manuscript_vla_ko.md`, `paper/sections_vla/`, `paper/refs/` 참고문헌 90편)
- docs/28~31, TASK_057~060

## 2026-10-01
### Added
- `vcp.features`(v1 호환/v2 FPS 불변 특징), `vcp.sim`(물리 기반 주행·AMR 합성 데이터), `vcp.experiments`(자동 실험·리포트), `vcp.vla`(언어 서술·큐레이션·LeRobot 구조 내보내기)
- comma.ai speedchallenge 실주행 데이터 수집/검출/라벨 도구
- `scripts/run_paper_suite.sh`, `scripts/build_paper.py`, 논문 원고 템플릿
- docs/25~27, TASK_050~056

### Changed
- `run_final_model.py` CLI/headless화, device 자동 선택, conf 기본값 0.45
- temporal 로더의 조용한 mock 대체를 설정으로 제어, 문헌형 체크포인트 로딩 지원
- 하이브리드 룰 게이트 특징 key 이름 기반 조회
- `.gitignore` 재설계(요약 결과와 원고만 추적)

### Removed
- 잘못 커밋된 `.venv_readme_check/`

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

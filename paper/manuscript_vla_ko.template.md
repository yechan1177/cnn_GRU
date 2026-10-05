# 수집 시점 맥락 인지 큐레이션과 소형 언어 조건 정책(VLA-lite)의 핵심 지표 기반 검증: 경량 CNN-GRU 점수기, 정책 블록 개선, 2단계 사전 등록 폐루프 실험

**Collection-Time Context-Aware Curation for a Small Language-Conditioned Policy (VLA-lite): KPI-Driven Validation with a Lightweight CNN-GRU Scorer, Policy-Block Improvements, and Two-Stage Preregistered Closed-Loop Experiments**

<!-- 생성기 안내: 이 원고는 `scripts/build_vla_paper.py`가 실험 결과(예비 연구 `experiments/exp_110_vla_curation/summary/`, 1차 확증 `experiments/exp_120_vla_v3/summary/`, 2차 확증 `experiments/exp_130_vla_v4/summary/`)로 자동 생성한다. 수치를 손으로 고치지 말고 템플릿(`paper/manuscript_vla_ko.template.md`, `paper/sections_vla/`)을 수정한다. -->

## 국문 초록

{{section:00_abstract_ko}}

## Abstract

{{section:00_abstract_en}}

**주제어**: 비전-언어-행동 모델, 데이터 큐레이션, 수집 시점 학습 데이터 선택, 핵심 지표, 사전 등록, 폐루프 평가, 위험 편향 붕괴, CNN-GRU, 피지컬 AI

{{section:01_intro_ko}}

{{section:02_related_work_ko}}

{{section:03_method_ko}}

{{section:04_setup_ko}}

{{section:05_results_ko}}

{{section:06_discussion_ko}}

{{section:07_conclusion_ko}}

## 데이터·코드 가용성

- 코드·설정·결과 요약은 공개 저장소에 있으며, 게재 확정 시 고정 태그와 영구 식별자(DOI)를 부여한다.
- 1차 확증(v3)은 `bash scripts/run_vla_v3.sh`, 2차 확증(v4, 개발 세트 선택과 새 테스트)은 `bash scripts/run_vla_v4.sh`, 예비 연구(v2)는 `bash scripts/run_vla_curation.sh`로 재현한다. KPI 표·그림은 `python -m vcp.experiments.vla_v3_report`와 `python -m vcp.experiments.vla_v4_report`, 원고는 `python scripts/build_vla_paper.py`로 다시 만든다(모두 `PYTHONPATH=src`).
- 실주행 데이터는 comma.ai speedchallenge 공개 데이터(`scripts/download_comma_speedchallenge.sh`)이다.
- 사전 등록 문서는 두 단계 모두 해당 테스트 실행 전에 저장소 커밋으로 고정했다.
  - 1차 확증(v3): `docs/34_v3_사전등록.md`, 커밋 `bee8b95`(2026-10-04 11:52 UTC). 등록 후 변경은 문서 4절에 시각과 함께 기록했다(판정 규칙은 바꾸지 않음).
  - 2차 확증(v4): `docs/37_v4_사전등록.md`, 커밋 `6f97127`(2026-10-04 23:26:29 UTC, 새 테스트 첫 단계 시작 23:26:32 UTC보다 앞섬). 등록 후 변경은 없으며, 이후 수정은 머리말의 확정 시각을 실제 커밋 시각으로 고친 한 줄(커밋 `1d57433`)뿐이다.
  - 무수정 여부는 `git diff bee8b95 -- docs/34_v3_사전등록.md`, `git diff 6f97127 -- docs/37_v4_사전등록.md`로 확인할 수 있다. 결과 기록은 각 계획 문서(`docs/33`, `docs/36`)의 실행 기록 절에 추가했다.
  - 예비 연구의 확증 실험 사전 등록은 `docs/32_확증실험_사전등록.md`(커밋 `fcd7f3f`)이다.

## 감사의 글

계층 부트스트랩 재분석과 AMR 성공 기준 점검을 제안해 준 심사위원들께 감사한다. 본문의 해당 수치는 모두 저자 구현(`src/vcp/experiments/vla_report.py`, `vla_v3_report.py`, `vla_v4_report.py`)으로 계산했다.

## 참고문헌

{{section:refs_list}}

# 수집 시점 큐레이션의 핵심 지표 기반 4단계 사전 등록 폐루프 검증: 공유 저장소 비교 설계와 소형 언어 조건 정책(VLA-lite)에서 얻은 교훈

**KPI-Driven Four-Stage Preregistered Closed-Loop Validation of Collection-Time Curation: A Shared-Reservoir Comparison Design and Lessons from a Small Language-Conditioned Policy (VLA-lite)**

<!-- 생성기 안내: 이 원고는 `scripts/build_vla_paper.py`가 실험 결과(예비 연구 `experiments/exp_110_vla_curation/summary/`, 1차 확증 `experiments/exp_120_vla_v3/summary/`, 2차 확증, 3차 확증(세 번째 테스트)과 4단계(새 학습 시드·네 번째 평가 세트) `experiments/exp_130_vla_v4/summary/`)로 자동 생성한다. 수치를 손으로 고치지 말고 템플릿(`paper/manuscript_vla_ko.template.md`, `paper/sections_vla/`)을 수정한다. -->

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
- 1차 확증(v3)은 `bash scripts/run_vla_v3.sh`, 2차 확증(v4, 개발 세트 선택과 새 테스트)은 `bash scripts/run_vla_v4.sh`, 예비 연구(v2)는 `bash scripts/run_vla_curation.sh`로 재현한다. 3차 심사 뒤의 탐색적 절제(새 테스트 재사용)는 `python -m vcp.experiments.vla_v4_suite explore`로, 3차 확증(세 번째 테스트)은 `bash scripts/run_vla_v4.sh third robot3 commap5 thirdreport`로, 4단계(새 학습 시드·네 번째 평가 세트)와 v4 + P5 폐루프 탐색은 `bash scripts/run_vla_v4.sh fourth fourthreport p5dev`로 재현한다(결과 `summary/fourth_kpi.json`; 탐색 요약 `summary/p5dev_explore.json`은 `p5dev` 실행 기록을 `vla_v4_suite.success_of`로 시드별 집계한 값이다). 4차 심사 대응 보조 분석(차이의 차이, 계층 구간, 시나리오 표, 2·3단계 종합, 같은 정책 대조, AMR 표본 계산)은 `python scripts/supp_round4_analysis.py`로 다시 계산한다(결과 `experiments/exp_130_vla_v4/summary/supp_round4.json`). 5차 심사 대응 보조 분석(4단계 시나리오 표, 4단계 K2 대응 구간, 4단계 차이의 차이, 정책 효과 − 선별 효과(사후), v4 + P5 탐색의 시드 3·4 비교와 새 실행 수, 4단계 고유 정책 수, 표 49의 한국어 판)은 `python scripts/supp_round5_analysis.py`로 다시 계산한다(결과 `summary/supp_round5.json`, 표 `supp5_*.md`). 그림 3은 `python -m vcp.experiments.vla_third_figure`로 만든다. KPI 표·그림은 `python -m vcp.experiments.vla_v3_report`와 `python -m vcp.experiments.vla_v4_report`, 원고는 `python scripts/build_vla_paper.py`로 다시 만든다(모두 `PYTHONPATH=src`).
- 실주행 데이터는 comma.ai speedchallenge 공개 데이터(`scripts/download_comma_speedchallenge.sh`)이다.
- 사전 등록 문서는 다섯 등록(예비 연구 확증과 네 확증) 모두 해당 테스트 실행 전에 저장소 커밋으로 고정했다.
  - 1차 확증(v3): `docs/34_v3_사전등록.md`, 커밋 `bee8b95`(2026-10-04 11:52 UTC). 등록 후 변경은 문서 4절에 시각과 함께 기록했다(판정 규칙은 바꾸지 않음).
  - 2차 확증(v4): `docs/37_v4_사전등록.md`, 커밋 `6f97127`(2026-10-04 23:26:29 UTC, 새 테스트 첫 단계 시작 23:26:32 UTC보다 앞섬). 등록 후 변경은 없으며, 이후 수정은 머리말의 확정 시각을 실제 커밋 시각으로 고친 한 줄(커밋 `1d57433`)뿐이다.
  - 3차 확증(세 번째 테스트): `docs/39_세번째테스트_사전등록.md`, 커밋 `e0ea4a5`(2026-10-06 16:09:21 UTC, 세 번째 테스트 첫 단계 시작 16:09:23 UTC보다 앞섬). 실행 코드와 분석 코드(`vla_third_report.py`)도 같은 커밋으로 고정했다. 등록 후 변경은 없으며, 이후 수정은 머리말의 확정 시각 기입(커밋 `84eb8af`)과 6절 결과 기록뿐이다.
  - 4단계(새 학습 시드·네 번째 평가 세트): `docs/40_네번째평가_새학습시드_사전등록.md`, 커밋 `6eff953`(2026-10-07 00:41:23 UTC, 4단계 첫 단계 시작 00:41:25 UTC보다 앞섬). 실행 코드(`vla_v4_suite.stage_fourth`·`stage_p5dev`)와 분석 코드(`vla_third_report.build_third_report`의 평가 세트 인자)도 같은 커밋으로 고정했다. 등록 문서 1~4절(설정·실행 조건·가설·해석 규칙)은 바뀌지 않았고, 이후 문서 수정은 6절 결과 기록(커밋 `96e4ad3`)과 5절 '등록 후 변경'의 기록(5차 심사 대응)이다. 등록 후 분석 코드의 변경은 반사실 표 머리글의 평가 세트 이름 표기 한 줄(커밋 `1788a5e`, 판정 로직 무관)이며, 이 변경과 변경 뒤의 분석 재실행(값 동일)을 docs/40 5절에 기록했다.
  - 무수정 여부는 `git diff bee8b95 -- docs/34_v3_사전등록.md`, `git diff 6f97127 -- docs/37_v4_사전등록.md`, `git diff e0ea4a5 -- docs/39_세번째테스트_사전등록.md`(1~5절), `git diff 6eff953 -- docs/40_네번째평가_새학습시드_사전등록.md`(1~4절; 5절은 등록 후 변경의 기록)로 확인할 수 있다. 결과 기록은 각 계획 문서(`docs/33`, `docs/36`)의 실행 기록 절에 추가했다.
  - 예비 연구의 확증 실험 사전 등록은 `docs/32_확증실험_사전등록.md`(커밋 `fcd7f3f`)이다.
  - 2차 확증의 사전 등록 초안은 커밋 `275b6fc`(2026-10-04 22:15:22 UTC)이며, 가설과 판정 규칙(3절)은 확정본과 같다(`git diff 275b6fc 6f97127 -- docs/37_v4_사전등록.md`).

## 부록 S. 본문이 인용하는 내부 문서와 측정 기록

본문의 `docs/…` 표기와 커밋 해시는 저자 저장소의 문서를 가리킨다. 저널 독자가 따라갈 수 있도록 아래에 묶었으며, 게재 시 저장소 태그와 함께 보관한다.

- **S1. 사전 등록 문서.** `docs/32_확증실험_사전등록.md`(예비 연구, `fcd7f3f`), `docs/34_v3_사전등록.md`(1단계, `bee8b95`), `docs/37_v4_사전등록.md`(2단계, 초안 `275b6fc`, 확정 `6f97127`, 머리말 시각 수정 `1d57433`), `docs/39_세번째테스트_사전등록.md`(3단계, `e0ea4a5`, 머리말 시각 기입 `84eb8af`, 6절 결과 기록), `docs/40_네번째평가_새학습시드_사전등록.md`(4단계, `6eff953`, 6절 결과 기록 `96e4ad3`, 5절 등록 후 변경 기록은 5차 심사 대응).
- **S2. 계획·실행 기록.** `docs/33`(1단계 계획과 실행 기록), `docs/36`(2단계 진단, 블록 정의, 채택 규칙, 실행 기록), `docs/38_3차심사_대응_탐색적_절제.md`(3차 심사 뒤 재분석과 탐색적 절제의 계획·결과), 3단계 결과 `experiments/exp_130_vla_v4/summary/third_kpi.json`, 4차 심사 대응 보조 분석 `experiments/exp_130_vla_v4/summary/supp_round4.json`(스크립트 `scripts/supp_round4_analysis.py`, 표 `supp4_*.md`), 5차 심사 대응 보조 분석 `summary/supp_round5.json`(스크립트 `scripts/supp_round5_analysis.py`, 표 `supp5_*.md`), 4단계 결과 `experiments/exp_130_vla_v4/summary/fourth_kpi.json`(표 `fourth_*.md`)과 v4 + P5 폐루프 탐색 `summary/p5dev_explore.json`, 단계 로그 `experiments/exp_130_vla_v4/logs/stages.log`, 3단계 중간 열람 커밋 `96e4c1c`(4.9.3절).
- **S3. 측정 기록.** 벤치마크 수정 전후의 전문가 측정 `experiments/exp_200_vla_sim/v3/a10_measurements.json`(스크립트 `measure_v3.py`), S1 점수 몫 재계산 `experiments/exp_121_v4_a15_obs_curation/summary/a15_measure.json`(스크립트 `scripts/measure_v4_a15.py`), 정책 학습 처리량 통제 측정 `experiments/exp_130_vla_v4/summary/a14_policy_blocks/bench_1thread.json`, 1단계 원자료 진단 수치(`v4_kpi.json`의 `v3_diag` 키, `vla_v4_report._v3_diagnosis`).

## 감사의 글

계층 부트스트랩 재분석과 AMR 성공 기준 점검을 제안해 준 심사위원들께 감사한다. 본문의 해당 수치는 모두 저자 구현(`src/vcp/experiments/vla_report.py`, `vla_v3_report.py`, `vla_v4_report.py`, `vla_third_report.py`, 4차·5차 심사 대응 보조 분석 `scripts/supp_round4_analysis.py`·`scripts/supp_round5_analysis.py`)으로 계산했다. 심사위원의 독립 재계산(차이의 차이, 계층 구간, 시나리오별 성공률, 2·3단계 종합 값, 4단계 K2 대응 구간, 정책 효과 − 선별 효과)은 저자 구현의 값과 대조했다.

## 참고문헌

{{section:refs_list}}

# 수집 시점 맥락 인지 큐레이션이 소형 언어 조건 정책(VLA-lite)의 종방향 제어에 미치는 효과: 경량 CNN-GRU 점수기와 폐루프 시뮬레이터 검증

**Collection-Time Context-Aware Data Curation for a Small Language-Conditioned Policy (VLA-lite) in Longitudinal Control: A Lightweight CNN-GRU Scorer Validated in Closed-Loop Simulation**

<!-- 생성기 안내: 이 원고는 `scripts/build_vla_paper.py`가 실험 결과(`experiments/exp_110_vla_curation/summary/`)로 자동 생성한다. 수치를 손으로 고치지 말고 템플릿(`paper/manuscript_vla_ko.template.md`, `paper/sections_vla/`)을 수정한다. -->

## 국문 초록

{{section:00_abstract_ko}}

## Abstract

{{section:00_abstract_en}}

**주제어**: 비전-언어-행동 모델, 데이터 큐레이션, 수집 시점 학습 데이터 선택, 폐루프 평가, 위험 편향 붕괴, 충돌시간, CNN-GRU, 피지컬 AI

{{section:01_intro_ko}}

{{section:02_related_work_ko}}

{{section:03_method_ko}}

{{section:04_setup_ko}}

{{section:05_results_ko}}

{{section:06_discussion_ko}}

{{section:07_conclusion_ko}}

## 데이터·코드 가용성

- 코드·설정·결과 요약은 공개 저장소에 있으며, 게재 확정 시 고정 태그와 영구 식별자(DOI)를 부여한다.
- 실험은 `bash scripts/run_vla_curation.sh`, 원고는 `python scripts/build_vla_paper.py`로 재현한다.
- 실주행 데이터는 comma.ai speedchallenge 공개 데이터(`scripts/download_comma_speedchallenge.sh`)이다.
- 확증 실험의 사전 등록 문서는 `docs/32_확증실험_사전등록.md`이며, 확증 실험 실행 전에 작성했다. 이 문서는 저장소 커밋 `fcd7f3f`(2026-10-02 04:56 UTC)로 고정했으며, 확증 결과는 이후 커밋에서 문서의 4절에만 추가했다(1~3절의 무수정 여부는 `git diff fcd7f3f -- docs/32_확증실험_사전등록.md`로 확인할 수 있다).

## 감사의 글

계층 부트스트랩 재분석과 AMR 성공 기준 점검을 제안해 준 심사위원들께 감사한다. 본문의 해당 수치는 모두 저자 구현(`src/vcp/experiments/vla_report.py`)으로 다시 계산했다.

## 참고문헌

{{section:refs_list}}

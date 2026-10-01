# 수집 시점 맥락 인지 큐레이션이 VLA 정책의 희소 위험 대응에 미치는 효과: 경량 CNN-GRU 점수기와 폐루프 검증

**Context-Aware, Risk-Enriched On-Device Data Curation for Vision-Language-Action Policies: A Lightweight CNN-GRU Scorer Validated by Closed-Loop Control**

> 이 원고는 `scripts/build_vla_paper.py`가 실험 결과(`experiments/exp_110_vla_curation/summary/`)로 자동 생성합니다. 수치를 손으로 고치지 말고 템플릿(`paper/manuscript_vla_ko.template.md`, `paper/sections_vla/`)을 수정하세요.

## 국문 초록

{{section:00_abstract_ko}}

## Abstract

{{section:00_abstract_en}}

**주제어**: 비전-언어-행동 모델, 데이터 큐레이션, 온디바이스 학습 데이터 선택, 폐루프 평가, 충돌시간, CNN-GRU, 피지컬 AI

{{section:01_intro_ko}}

{{section:02_related_work_ko}}

{{section:03_method_ko}}

{{section:04_setup_ko}}

{{section:05_results_ko}}

{{section:06_discussion_ko}}

{{section:07_conclusion_ko}}

## 데이터·코드 가용성

- 코드·설정·결과 요약은 저장소 브랜치 `claude/magical-faraday-8wqdto`에 있습니다.
- 실험은 `bash scripts/run_vla_curation.sh`, 원고는 `python scripts/build_vla_paper.py`로 재현합니다.
- 실주행 데이터는 comma.ai speedchallenge 공개 데이터(`scripts/download_comma_speedchallenge.sh`)입니다.

## 참고문헌

{{section:refs_list}}

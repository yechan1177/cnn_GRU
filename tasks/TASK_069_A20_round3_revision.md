# TASK_069 (A20) 3차 심사(Major) 대응 원고 개정

근거: `paper/review/review_round3_ko.md`(M1~M12, m1~m20, A1~A8), 대응 방침 `docs/38_3차심사_대응_탐색적_절제.md`, 결과 `experiments/exp_130_vla_v4/summary/v4_kpi.json`·`tables/`(v3는 `experiments/exp_120_vla_v3/summary/`).
범위: `paper/manuscript_vla_ko.template.md`, `paper/sections_vla/00~07_*.md`, `src/vcp/experiments/vla_v4_report.py`, `scripts/build_vla_paper.py`, 답변서 `paper/review/response_round3_ko.md`, `docs/35`·`docs/36`(m14)·`docs/38`(4절).
제외: 재실험(A4~A7), git commit/push(본 세션이 한다).
원칙: 사전 등록 판정을 바꾸지 않는다(해석만 보탬). 탐색적 결과에는 '탐색적(새 테스트 재사용)'을 붙인다. 미달은 미달.

## 체크리스트
- [x] 입력 확인: 3차 심사, docs/38, v4_kpi.json 새 키, 표 v4_driving_scenario·v4_explore_k6·v4_explore_k4, 원고 원천·빌더
- [x] 보고서 보강(`vla_v4_report.py`, 기존 키 값 불변을 키 단위로 확인)
  - [x] explore.k6_alias(영문 별칭: full_lang, full_nolang, p6_lang, p6_nolang, p67_only, ctrl)와 시드별 값·CI·성공률·자유주행 프레임, explore.k6_learned_lang_path
  - [x] 위험 시나리오 시드 SD(hazard_sd, 표 v4_driving), K3 margin·margin_episodes, driving_scenario, expert_test_success(hazard·vru_crossing·robot), expert_cf
  - [x] K6 성공률·프레임·sd_seed_reduction, K5 검정력(by_seed, sd_seed_diff, halfwidth, seeds_for_halfwidth_0.03, v3_halfwidth), ratios
  - [x] v3_diag(1단계 진단 수치 재계산: AMR 27/36·15/36·속도·개루프 MAE, 반사실 스타일별 속도, 실영상 점수기 AUROC 0.518·−a_t 0.802, S1 점수 몫 구성) — 원고 값과 일치 확인
  - [x] 표 한국어화(v4_dev_lang, v4_explore_k4, v4_explore_k6, v4_driving_scenario, v4_language_cf, v4_kpi 보조 구간, v4_driving의 탐색 행 표시)
  - [x] 그림 3: 새 테스트의 v3 설정 막대 추가, 교차 세트 구분
  - [x] K4 판정 함수에 Holm 조건 추가(m13, 판정 불변)
- [x] 빌더: `{{u:...}}`(부호 없는 값) 추가, p·pp 누락 보고
- [x] 초록(국·영): M1·M4·M5·M6·M10·m20 반영
- [x] 제목 완화(M10), 서론 기여 재배열·표 1 개정(M1·M6·M10·m19), 2.10절(m9·m10)
- [x] 3절: 3.4 트리거 혼합 단서, 3.6.2 가설 표현·P5 탈락(m18)·S1 수치 키, 3.6.3 벤치마크 진단 표 이동(M11), 3.6.4 라벨 정의(M8), 3.7 K4 추정 대상(M3)·K6 평균의 비와 프레임 선택 효과(m4), 표 8 캡션 K7 예외(M8)
- [x] 4절 재배치(M11): 변수 → 측정·통제 → 환경 → 세팅·1단계 → 사전 등록 → 2단계 개발 → 2단계 확증 → 탐색적 절제 설계(표 20), 표 18에 275b6fc(M6·m12), 검정력 예견(M7), 부록 S 인용(m11)
- [x] 5절: 5.1 M1 문장·그림 3 캡션(M9), 5.3 수치 키화(M12), 5.4 개발 결과 일원화·K6 8.8%(M2), 5.5 K1~K3 대비·K3 경계·시나리오 표·충돌률 순서·p 표기(M1·M5·m16·m17), K6 시드 쌍봉, AMR 교차 세트(M7), K7 라벨 근접성(M8), 5.6 탐색적 절제 신설(표 37·38), 5.7 비용 출처(M12·m7), 5.8 예비 연구 통합
- [x] 6절: 6.1 교차 세트·P7 가능성(M6·M9), 6.2 제목·동등성 삭제·추정 대상(M3·M4), 6.3 예견 가능한 검정력(M7), 6.4 가설로 하향(M8), 6.5 메커니즘(M2), 6.6 KPI 체계 한계·하지 않은 재실험(M1), 6.7 권고 약화와 근거 등급(M4)
- [x] 7절: 표 40 재작성(H-v 행, 교차 세트, 조건부 K6, 경계 K3), 해결 정도·한계·향후 연구(세 번째 세트·새 사전 등록)
- [x] 템플릿: 데이터·코드 가용성(탐색 재현, 초안 커밋), 부록 S(내부 문서·측정 기록)
- [x] 표 1~40·그림 1~4 연속 번호, 모든 표 캡션 앞 인용, 절 참조·조사 점검(스크립트)
- [x] 빌드 누락 0, `nan`·`{{`·`[누락` 0
- [x] 답변서 `paper/review/response_round3_ko.md`
- [x] docs/36 m14 정정, docs/38 4절(반영 결과)·해석 보정, docs/35 5절(3차 심사 반영)

## 메모
- 탐색적 K6 분해에서 'v4 − P6·P7만' 차이(+0.47 m/s)는 시드 대응 95% 구간 [−0.10, +1.04]로 0을 포함한다. docs/38의 "학습 언어 경로가 오차를 늘린다"는 원고에서 "더 줄였다는 증거가 없다"로 완화했다.
- "12~14 m/s"(옛 5.3절)는 결과 파일로 재현되지 않아, 재계산한 에피소드 평균 속도(10.7~11.6 m/s)와 자유주행 오차(2.4~4.6 m/s)로 바꿨다.
- 남은 손 수치: 표 5(전문가 측정, `a10_measurements.json`), 표 14·15(파일럿·격자, `pilot.log`), 표 39 통제 처리량(`bench_1thread.json`) — 모두 출처 파일을 캡션에 적었다.

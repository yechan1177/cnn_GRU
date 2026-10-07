## 5. 실험 결과

결과는 연구 단계 순서로 보고한다. 5.1절은 핵심 지표(KPI) K1~K7의 판정을 네 확증 단계에 걸쳐 요약한다. 5.2절은 1차 확증(v3, 사전 등록 docs/34, 테스트 시드 800000번대)의 결과다. 5.3절은 v3 원자료로 미달 원인을 진단한 사후 분석이고, 5.4절은 그 진단에 따라 정책·학습 블록을 고른 v4 개발 세트 결과다. 5.5절은 처음 보는 새 테스트 세트(시드 1000000번대)에서 수행한 2차 확증(v4, 사전 등록 docs/37)의 결과다. 5.6절은 3차 심사 뒤 새 테스트 세트를 재사용한 탐색적 절제이고, 5.7절은 그 탐색이 세운 가설과 심사 요구 분석을 처음 보는 세 번째 테스트 세트(시드 1200000번대)에서 확증한 3차 확증(사전 등록 docs/39)의 결과다. 3차 확증의 정책은 대부분 2단계와 같은 학습 정책이다(4.9.1절). 5.8절은 새 학습 시드(10~19)로 새로 학습한 정책을 처음 보는 네 번째 평가 세트(시드 1400000번대)에서 평가한 4단계 독립 반복(사전 등록 docs/40)의 결과다. 5.9절은 계산 비용, 5.10절은 예비 연구(v2)의 설계와 결과를 요약한다.

확증 근거는 5.2절, 5.5절, 5.7절(실영상 P5 제외), 5.8절(개발 세트의 v4 + P5 탐색 제외)뿐이다. 5.3·5.4절은 v3 테스트 결과를 본 뒤의 분석과 선택이고, 5.6절은 v4 새 테스트 결과를 본 뒤의 탐색이므로 모두 탐색적 결과로 읽어야 한다. 판정 규칙은 3.7절(표 8)과 같으며, 점추정 기준 지표(K1~K3, K6, K7)는 점추정이 목표 이상이면 '달성'이다.

### 5.1 핵심 지표 요약

표 23은 네 확증 단계의 핵심 지표 값과 사전 등록 판정이고, 그림 3은 비율형 지표를 같은 세트 비교와 교차 세트 비교로 나누어 그린 것이다.

| ID | 지표 | 목표 | 1차(v3 설정, v3 테스트 800000번대) | 2차(v4 설정, 새 테스트 1000000번대) [95% 구간] | 3차(v4 설정, 세 번째 테스트 1200000번대) [95% 구간] | 4단계(v4 설정, 새 학습 시드 10~19, 네 번째 평가 1400000번대) [95% 구간] | 판정(1차 / 2차 / 3차 / 4단계) |
|---|---|---|---|---|---|---|---|
| K1 | CARE 2% 폐루프 성공률 | ≥ 0.80 | {{s:k:K1:value}} | {{s:k4:K1:value}} [{{u:k4:K1:lo}}, {{u:k4:K1:hi}}] | {{s:k3:kpi:K1:value}} [{{u:k3:kpi:K1:lo}}, {{u:k3:kpi:K1:hi}}] | {{s:k5:kpi:K1:value}} [{{u:k5:kpi:K1:lo}}, {{u:k5:kpi:K1:hi}}] | 미달 / 달성 / 달성 / 달성 |
| K2 | 데이터 효율(CARE 2% / 전체) | ≥ 0.90 | {{s:k:K2:value}} | {{s:k4:K2:value}} [{{u:k4:K2:lo}}, {{u:k4:K2:hi}}] | {{s:k3:kpi:K2:value}} [{{u:k3:kpi:K2:lo}}, {{u:k3:kpi:K2:hi}}] | {{s:k5:kpi:K2:value}} [{{u:k5:kpi:K2:lo}}, {{u:k5:kpi:K2:hi}}] | 미달 / 미달 / 달성 / 달성 |
| K3 | 위험 시나리오 성공률(CARE 2%) | ≥ 0.75 | {{s:k:K3:value}} | {{s:k4:K3:value}} [{{u:k4:K3:lo}}, {{u:k4:K3:hi}}] | {{s:k3:kpi:K3:value}} [{{u:k3:kpi:K3:lo}}, {{u:k3:kpi:K3:hi}}] | {{s:k5:kpi:K3:value}} [{{u:k5:kpi:K3:lo}}, {{u:k5:kpi:K3:hi}}] | 미달 / 달성 / 달성 / 달성 |
| K4 | 점수기 고유 기여(CARE − 트리거 혼합) | CI 하한 > 0 | {{s:k:K4:diff}} [{{s:k:K4:lo}}, {{s:k:K4:hi}}] | {{s:k4:K4:diff}} [{{s:k4:K4:lo}}, {{s:k4:K4:hi}}] | {{s:k3:tests:K4:diff}} [{{s:k3:tests:K4:lo}}, {{s:k3:tests:K4:hi}}] | {{s:k5:tests:K4:diff}} [{{s:k5:tests:K4:lo}}, {{s:k5:tests:K4:hi}}] | 미달 / 미달 / 미달 / 미달 |
| K5 | AMR CARE − 무작위(공유 저장소) | CI 하한 > −0.03 | {{s:k:K5:diff}} [{{s:k:K5:lo}}, {{s:k:K5:hi}}] | {{s:k4:K5:diff}} [{{s:k4:K5:lo}}, {{s:k4:K5:hi}}] | {{s:k3:robot:K5:diff}} [{{s:k3:robot:K5:lo}}, {{s:k3:robot:K5:hi}}](시드 10) | —(AMR 미실시) | 미달 / 미달 / 미달 / 판정 없음 |
| K6 | 언어가 줄이는 속도 추종 오차(반사실) | ≥ 30% | {{p:k:K6:speed_error_reduction}} | {{p:k4:K6:speed_error_reduction}} [{{p:k4:K6:lo}}, {{p:k4:K6:hi}}]; 계층 [{{p:s4:k6_hier_stage2:lo}}, {{p:s4:k6_hier_stage2:hi}}] | {{p:k3:k6:K6:speed_error_reduction}} [{{p:k3:k6:K6:lo}}, {{p:k3:k6:K6:hi}}]; 계층 [{{p:k3:k6:K6_hier:lo}}, {{p:k3:k6:K6_hier:hi}}] | {{p:k5:k6:K6:speed_error_reduction}} [{{p:k5:k6:K6:lo}}, {{p:k5:k6:K6:hi}}]; 계층 [{{p:k5:k6:K6_hier:lo}}, {{p:k5:k6:K6_hier:hi}}] | 미달 / 달성 / 달성 / 달성 |
| K7 | 실영상 제동 시작 AUROC(전체 데이터) | ≥ 0.65 | {{s:k:K7:value}} | {{s:k4:K7:value}} | —(새 실영상 없음. v4+P5 탐색 {{s:k3:comma_p5:auroc_p5}}) | —(실영상 미실시) | 미달 / 미달 / 판정 없음 / 판정 없음 |

표 23. 핵심 지표의 목표와 판정. 1차 열은 1차 확증(사전 등록 docs/34), 2차 열은 2차 확증(사전 등록 docs/37), 3차 열은 3차 확증(사전 등록 docs/39), 4단계 열은 새 학습 시드의 독립 반복(사전 등록 docs/40)의 값이다. 네 열은 평가 세트가 서로 다르므로(교차 세트) 열 사이의 차이를 개선 폭으로 읽지 않는다. 특히 2차와 3차는 학습 시드까지 같은 같은 정책(4.9.1절)인데도 값이 다르며, 이 차이는 평가 세트의 난이도와 에피소드 표집에서만 온다(5.7절 (7)). 4단계의 정책은 학습 시드 10~19로 새로 학습했으므로 3차와 4단계의 차이에는 평가 세트와 학습 시드 표집이 함께 작용한다. 같은 세트에서 v3·v4 설정을 직접 비교한 값은 2차의 H-v(표 35), 3차의 H-v3(표 43), 4단계의 H-v4(표 48)다. K1~K3의 대괄호는 시드·에피소드 부트스트랩 95% 구간, K6의 첫 대괄호는 시드 부트스트랩 95% 구간, '계층'은 시드·에피소드 계층 부트스트랩 95% 구간(4차 심사 대응 보조)으로, 모두 판정에는 쓰지 않는 보조 정보다. K2의 구간은 CARE와 전체 데이터를 독립으로 재표집한 보수적 구간이다(4.2절). K4·K5는 시드 대응 차이와 시드·에피소드 계층 부트스트랩 95% CI다. 2·3차·4단계의 K4는 1차의 K4와 추정 대상이 다르다(3.7절). 2차의 K7은 1단계와 같은 fold를 재사용한 값으로 새 데이터의 확증이 아니며(4.7.1절), 3차에는 새 실영상이 없어 K7을 판정하지 않았다(4.9.1절). 4단계는 사전 등록대로 AMR과 실영상을 넣지 않아 K5·K7을 판정하지 않았다(4.10절).

{{fig:fig_third_kpi.png|그림 3. 비율형 핵심 지표와 목표(붉은 파선). 빗금 막대는 1차(v3 설정, v3 테스트), 밝은 파란색은 2차(v4 설정, 새 테스트), 진한 파란색은 3차(v4 설정, 세 번째 테스트)의 값으로, 주황색은 4단계(v4 설정, 새 학습 시드 10~19, 네 번째 평가 세트)의 값으로, 네 막대는 평가 세트가 서로 다르다(교차 세트). 검은 ×는 같은 세트에서 잰 v3 설정의 값(K1·K3, 2차·3차·4단계 막대 위)이며, 같은 세트의 대응 비교(× 대 막대)만 개선의 근거다. 그 검정은 표 35의 H-v, 표 43의 H-v3, 표 48의 H-v4다. K6은 언어 입력이 줄인 속도 추종 오차의 비율이며, v3에서는 음수(언어가 오차를 키움)였다. 2차·3차·4단계 막대의 오차 막대는 95% 구간이다(K1~K3: 시드·에피소드 부트스트랩, K6: 시드·에피소드 계층 부트스트랩, 보조). 1차와 K7은 결과 파일에 구간이 없어 오차 막대가 없다. K7의 초록 점선은 같은 시험 프레임에서 정책 없이 현재 자차 가속도만 쓴 $-a_t$ 기준선(값은 5.5절 (5))으로, 라벨 정의에 가까운 참고값이다. 빈 마름모는 3차에서 자차 운동 이력(P5)을 넣은 v4 정책의 실영상 AUROC로, 같은 fold를 재사용한 탐색적 값이며 목표선을 넘었지만 $-a_t$ 기준선 아래다(K7은 3차에서 판정하지 않음). 4단계는 실영상이 없어 K7 막대가 없다. 차이형 지표 K4·K5는 표 23에 있다.}}

- **1차 확증(v3)에서는 일곱 지표가 모두 미달했다.** 성공률은 목표에 가까웠으나(K1 {{s:k:K1:value}}, 목표 0.80), 점수기 고유 기여(K4)는 CI 하한이 0에 닿지 못했고, AMR 비열등(K5)·언어 지시 준수(K6)·실영상 제동 시작 예측(K7)은 목표와 거리가 멀었다.
- **2차 확증(v4)에서는 사전 등록 점추정 기준으로 K1·K3·K6을 달성했고, K2·K4·K5·K7은 미달했다.** 값은 K1 {{s:k4:K1:value}}, K3 {{s:k4:K3:value}}, K6 {{p:k4:K6:speed_error_reduction}}이다. K3은 목표와의 차이가 {{u:k4:K3:margin}}에 불과한 경계 달성이고, K6은 시드별 감소가 고르지 않았다(5.5절). K2({{s:k4:K2:value}})는 목표 0.90에 조금 못 미쳤고, K4({{s:k4:K4:diff}})는 사실상 0이었다.
- **K1·K3의 달성은 선별법을 판별하지 않는다.** 같은 새 테스트의 v4 정책에서 무작위 2%({{s:k4:driving:v4:random_shared:success}})도 K1 목표를 넘었고, 감속 트리거 혼합은 K1({{s:k4:driving:v4:mix_trigger:success}})과 K3({{s:k4:driving:v4:mix_trigger:hazard}})를 모두 넘었다(표 33). 따라서 K1·K3 달성은 CARE 선별이 아니라 정책 설정 변경의 성과이며, CARE 선별의 고유 효과는 확인되지 않았다.
- v4의 개선은 큐레이션(점수기·풀·선별 하이퍼파라미터)을 그대로 두고 하위 정책과 학습 블록만 바꾼 결과다(3.6절). 같은 새 테스트에서 v4 설정은 v3 설정보다 CARE 2% 성공률이 {{s:k4:comparisons:v4_vs_v3_care:diff}} 높았다(5.5절, H-v).
- **3차 확증(세 번째 테스트)에서는 K1·K2·K3·K6을 달성했고, K4·K5는 미달했다.** 값은 K1 {{s:k3:kpi:K1:value}}, K2 {{s:k3:kpi:K2:value}}, K3 {{s:k3:kpi:K3:value}}, K6 {{p:k3:k6:K6:speed_error_reduction}}(시드 {{s:k3:k6:K6:n_seeds:0}}개 중 {{s:k3:k6:K6:n_seeds_reduced:0}}개에서 감소)이다. K2는 2차의 미달에서 달성으로 바뀌었고, K3·K6의 여유도 커졌다. 3차의 정책은 2차와 같은 학습 정책이므로(4.9.1절) 이 차이는 평가 세트에서만 온다. 같은 정책 넷(v4 CARE·트리거 혼합·무작위, v3 CARE)과 전문가의 성공률이 세 번째 세트에서 함께 높았으므로(예: v3 CARE {{s:k4:driving:v3:care:success}} → {{s:k3:tests:H-v3:b}}, 전문가 {{s:s4:expert:stage2:driving}} → {{s:s4:expert:stage3:driving}}), 세 번째 세트가 더 쉬웠다. 사전 등록 규칙(docs/39 4절)에 따라 2차와 3차 결과를 모두 보고하며, 두 세트의 에피소드를 합친 K2는 {{s:s4:pooled_23:K2:value}}(목표 이상 비율 {{p:s4:pooled_23:K2:p_ge_goal:0}})로 목표와 거의 같았다(5.7절 (7)).
- **3차의 확증 가설(5.7절)은 다음과 같다.** v4 설정의 개선(H-v3)과 v4의 CARE − 무작위(H-a3)는 지지되었다. v4 정책에서 점수기 출력만 학습에서 뺐을 때의 CARE 선별 효과(H-a3n), P7의 성공률 기여(H-P7), 학습 언어 경로의 몫(H-L)은 지지되지 않았다. H-a3·H-a3n의 차이({{s:s4:ha_diff:diff}} [95% CI {{s:s4:ha_diff:lo}}, {{s:s4:ha_diff:hi}}], 보조)도 유의하지 않았으므로, 두 조건의 선별 효과가 다르다는 증거는 없다. 점수기 출력을 학습에 쓰지 않는 조건의 CARE − 트리거 혼합은 차이도 동등(±0.03)도 확정하지 못했다(H-K4e).
- **4단계(새 학습 시드 10~19, 네 번째 평가 세트)에서는 K1·K2·K3·K6을 달성했고 K4는 미달했다**(K5·K7은 실시하지 않음). 값은 K1 {{s:k5:kpi:K1:value}}, K2 {{s:k5:kpi:K2:value}}(목표 이상 비율 {{p:k5:kpi:K2:p_ge_goal:0}}), K3 {{s:k5:kpi:K3:value}}, K6 {{p:k5:k6:K6:speed_error_reduction}}(계층 구간 하한 {{p:k5:k6:K6_hier:lo}}, 목표 아래), K4 {{s:k5:tests:K4:diff}} [{{s:k5:tests:K4:lo}}, {{s:k5:tests:K4:hi}}]이다. 확증 가설은 H-v4·H-a4·H-a4n이 지지, H-P7b·H-L4가 미지지였고 H-K4e는 동등이 아니었다. 사전 등록 해석 규칙(docs/40 4절)에 따라 3단계와 판정이 같은 KPI·가설은 '독립 학습 시드에서 재현'으로, 판정이 달라진 H-a(noaux)(3단계 미지지 → 4단계 지지)는 '불안정'으로 쓴다(5.8절, 표 50). H-L4는 미지지였을 뿐 아니라 점추정이 반대 방향이었다(학습 언어 경로를 더한 쪽의 오차가 더 큼, {{s:k5:k6:H-L:diff:2}} m/s [{{s:k5:k6:H-L:lo:2}}, {{s:k5:k6:H-L:hi:2}}]).

### 5.2 1차 확증(v3)

1차 확증은 사전 등록(docs/34, 커밋 `bee8b95`)에 따라 주행 테스트 147 에피소드(시드 800000번대), 반사실 언어 세트 63 에피소드, AMR 테스트 72 에피소드, 실영상 5-fold × 시드 2개로 수행했다. CARE의 $\lambda$=0.5, $\rho$=0.9(AMR은 $\lambda$=0, $\rho$=0.95)는 테스트 전에 검증 세트에서 정했다(4.4.2절).

**(1) 주행 폐루프(K1~K3).** 표 24는 v3 주행 테스트 결과이고, 그림 4는 예산별 성공률이다.

{{table:v3_driving}}

표 24. v3 주행 테스트 결과(평균 ± 시드 표준편차). 예산 2%는 시드 10개, 1%·5%와 전체 데이터는 시드 5개다. 혼합 방법은 같은 학습 시드에서 공유 저장소(예산의 $\rho$)를 공유하고 점수 몫만 다르다(3.4절). *는 GT 위험 라벨을 쓰는 특권 정보 조건이다.

- **K1(미달)**: CARE 2%의 성공률({{s:k:K1:value}} ± {{s:k:K1:sd}})은 목표 0.80에 미달했다.
- **K2(미달)**: 전체 데이터의 성공률({{s:k:K2:full}})이 높아 비율은 {{s:k:K2:value}}에 그쳤다(목표 0.90).
- **K3(미달)**: 위험 시나리오 성공률은 {{s:k:K3:value}} ± {{s:k:K3:sd}}에 그쳤다(목표 0.75).
- 감속 트리거 단독 선택은 예비 연구(5.10절)와 같은 위험 편향 붕괴를 보였다. 성공률이 0.1 아래였고(표 24), 속도 오차가 커 진행 기준을 넘지 못했다. CARE − 감속 트리거 단독의 차이: {{s:k:comparisons:care_vs_trigger_only_b0.02:diff}}.

{{fig:fig_v3_budget.png|그림 4. v3 주행 테스트의 예산별 폐루프 성공률(CARE, 무작위(공유 저장소); 오차 막대는 시드 표준편차). 점선은 전체 데이터(100%), 붉은 파선은 K1 목표(0.80)다. 예산 1%·5%는 사전 등록상 보조(탐색적) 분석이다.}}

**(2) 점수기 고유 기여(K4)와 보조 가설.** 표 25는 사전 등록 가설 검정이다.

| 비교(예산 2%, 시드 10개 대응) | 차이 | 계층 부트스트랩 95% CI | 단측 $p$ | Holm 보정 $p$ | 판정 |
|---|---|---|---|---|---|
| K4: CARE − 저장소 + 감속 트리거 | {{s:k:K4:diff}} | [{{s:k:K4:lo}}, {{s:k:K4:hi}}] | {{s:k:K4:p_le0:3}} | {{s:k:K4:p_holm:3}} | 미달 |
| H-a: CARE − 무작위(공유 저장소) | {{s:k:comparisons:care_vs_random_shared_b0.02:diff}} | [{{s:k:comparisons:care_vs_random_shared_b0.02:lo}}, {{s:k:comparisons:care_vs_random_shared_b0.02:hi}}] | {{s:k:comparisons:care_vs_random_shared_b0.02:p_le0:4}} | {{s:k:comparisons:care_vs_random_shared_b0.02:p_holm:3}} | 지지 |
| H-b: CARE − 저장소 + 오라클* | {{s:k:comparisons:care_vs_mix_oracle_b0.02:diff}} | [{{s:k:comparisons:care_vs_mix_oracle_b0.02:lo}}, {{s:k:comparisons:care_vs_mix_oracle_b0.02:hi}}] | {{s:k:comparisons:care_vs_mix_oracle_b0.02:p_le0:3}} | {{s:k:comparisons:care_vs_mix_oracle_b0.02:p_holm:3}} | 차이 판별 안 됨 |
| (탐색) 저장소 + 감속 트리거 − 무작위 | {{s:k:comparisons:mix_trigger_vs_random_shared_b0.02:diff}} | [{{s:k:comparisons:mix_trigger_vs_random_shared_b0.02:lo}}, {{s:k:comparisons:mix_trigger_vs_random_shared_b0.02:hi}}] | {{s:k:comparisons:mix_trigger_vs_random_shared_b0.02:p_le0:3}} | 해당 없음 | - |
| (탐색) CARE − 무작위, 예산 1%(시드 5개) | {{s:k:comparisons:care_vs_random_shared_b0.01:diff}} | [{{s:k:comparisons:care_vs_random_shared_b0.01:lo}}, {{s:k:comparisons:care_vs_random_shared_b0.01:hi}}] | {{s:k:comparisons:care_vs_random_shared_b0.01:p_le0:4}} | 해당 없음 | - |
| (탐색) CARE − 무작위, 예산 5%(시드 5개) | {{s:k:comparisons:care_vs_random_shared_b0.05:diff}} | [{{s:k:comparisons:care_vs_random_shared_b0.05:lo}}, {{s:k:comparisons:care_vs_random_shared_b0.05:hi}}] | {{s:k:comparisons:care_vs_random_shared_b0.05:p_le0:4}} | 해당 없음 | - |

표 25. v3 사전 등록 가설 검정(docs/34 3절). 신뢰구간은 시드·에피소드 2단계 계층 부트스트랩(10,000회)이고, 단측 $p$는 차이 ≤ 0인 재표집의 비율이다. Holm 보정 군은 K4·H-a·H-b다. '(탐색)' 행은 사전 등록에서 보조 분석으로 정한 기술 통계이며 다중 비교 보정을 하지 않았다.

- **K4(미달)**: CARE는 같은 저장소에 감속 트리거 점수 몫을 더한 통제군보다 {{s:k:K4:diff}} 높았으나, 95% CI 하한이 0 아래({{s:k:K4:lo}})였고 Holm 보정 $p$도 0.05를 넘었다({{s:k:K4:p_holm:3}}). 시드 대응 $t$ 구간([{{s:k:K4:seed_t:lo}}, {{s:k:K4:seed_t:hi}}])은 0을 포함하지 않았지만, 사전 등록 판정은 계층 부트스트랩으로 한다.
- **H-a(지지)**: 공유 저장소를 고정하고 점수 몫만 무작위 몫과 바꾼 비교에서 CARE가 {{s:k:comparisons:care_vs_random_shared_b0.02:diff}} 높았다. 저장소 클립이 같으므로 이 차이는 예산의 10%인 점수 몫의 내용에서 나온다.
- 탐색 비교에서 예산 1%·5%의 CARE − 무작위도 양(+)이었고 CI가 0을 포함하지 않았다(그림 4). 감속 트리거 점수 몫의 무작위 대비 차이는 CI가 0을 포함했다. 즉 점수기 출력 없이 학습한 1단계에서는 '아무 위험 지향 몫'이 무작위 몫보다 낫다는 증거가 없었다.

**(3) 시나리오와 선택 구성(보조, 탐색적).** 표 26은 시나리오별 성공률, 표 27은 선택 데이터의 구성이다.

{{table:v3_driving_scenario}}

표 26. v3 시나리오별 성공률(예산 2%는 시드 10개, 전체는 시드 5개 평균). 칸별 신뢰구간은 계산하지 않았으며 탐색적 기술 통계다.

- CARE가 무작위보다 높았던 칸은 주로 위험 시나리오(선행차 급제동, 정체)와 밀집이었다. 추종에서는 무작위가 더 높았다(표 26).
- 보행자 횡단은 모든 2% 방법에서 0.5 아래였고, 전체 데이터에서도 다른 시나리오보다 낮았다. 이 시나리오의 낮은 성공률은 v4 진단(5.3절)의 근거가 되었다.

{{table:v3_selection}}

표 27. v3 예산 2% 선택 데이터의 구성(시드 평균). 위험 프레임 비중은 선택 프레임 가운데 GT 위험 프레임의 비율, 위험 이벤트 회수율은 풀의 위험 이벤트 가운데 선택에 포함된 비율, 에피소드 포괄률은 선택 클립이 나온 에피소드의 비율이다.

- 혼합 방법(CARE·감속 트리거·오라클)의 선택 전체 위험 비중은 0.12~0.15로 무작위(0.07)의 1.6~2배였고, 감속 트리거 단독은 0.6을 넘었다.
- 세 혼합 방법은 위험 비중·회수율·엔트로피·에피소드 포괄률이 서로 비슷했다. 이 구성 차이만으로는 K4의 차이를 설명하기 어렵다(점수 몫만의 구성은 5.3절 5).

**(4) 언어 지시 준수(K6).** 표 28은 반사실 언어 평가다.

{{table:v3_language_cf}}

표 28. v3 반사실 언어 평가(시드 5개). 같은 (시나리오, 시드)를 세 스타일 지시문으로 각각 평가했다. '신중−민첩 추종 간격 차'는 같은 (시나리오, 시드)에서 신중 지시와 민첩 지시의 평균 추종 간격 차이다.

- **K6(미달)**: CARE 2%에서 언어를 넣으면 속도 추종 오차가 {{s:k:K6:nolang:speed_error:2}} → {{s:k:K6:lang:speed_error:2}} m/s로 오히려 늘었다(감소율 {{p:k:K6:speed_error_reduction}}).
- 전체 데이터(보조)에서는 {{s:k:K6:by_data:full:nolang:speed_error:2}} → {{s:k:K6:by_data:full:lang:speed_error:2}} m/s로 {{p:k:K6:by_data:full:speed_error_reduction}} 줄었다.
- 스타일별 추종 간격 분리도는 두 데이터 모두 언어를 넣으면 커졌다(CARE {{s:k:K6:nolang:style_sep:2}} → {{s:k:K6:lang:style_sep:2}} s). 즉 2% 데이터의 정책은 지시문으로 추종 간격은 구분했지만 목표 속도는 따르지 못했다.

**(5) AMR(K5).** 표 29는 AMR 테스트 결과다.

{{table:v3_robot}}

표 29. v3 AMR 테스트 결과(72 에피소드; 2%는 시드 5개, 전체는 시드 3개). AMR 튜닝(검증, 시드 0)은 $\lambda$=0, $\rho$=0.95를 골랐다.

- **K5(미달)**: CARE − 무작위(공유 저장소)는 {{s:k:K5:diff}} [95% CI {{s:k:K5:lo}}, {{s:k:K5:hi}}]로, CI 하한이 비열등 한계 −0.03보다 훨씬 낮았다(표 29).
- 전체 데이터 정책의 성공률({{s:k:robot_full_success}})도 전문가({{s:k:robot_expert_success}})의 절반 수준이었다. AMR 과제는 데이터 선택 이전에 이 정책에게 어려웠다.

**(6) 실영상(K7).** 표 30은 실영상 개루프 결과다.

{{table:v3_comma}}

표 30. v3 comma.ai 실영상 개루프 결과(5-fold 블록 교차검증 × 시드 2개 = 10회 실행, ± 는 실행 간 표준편차). '특징 토큰'은 실제 YOLOv8n 검출 특징 입력 여부다. 제동 시작 AUROC는 현재 제동 중이 아닌 프레임에서 1초 안의 제동 시작을 판별한 값이다(3.7절).

- **K7(미달)**: 전체 데이터 특징 토큰 정책의 제동 시작 AUROC는 {{s:k:K7:value}} ± {{s:k:K7:sd}}에 그쳤다(특징 토큰 없음 {{s:k:K7:nofeat}}, 표 30). 예비 연구와 마찬가지로 우연 수준이었다.
- 예산 10·20%의 방법 간 AUROC도 0.51~0.58 범위였다. 감속 트리거 단독은 제동 구간 MAE가 가장 낮았지만 전체 MAE가 가장 높아, 예비 연구의 개루프 트레이드오프를 재현했다.

### 5.3 미달 원인 진단(v3 원자료, 사후 분석)

1차 확증의 미달 원인은 v3 원자료를 다시 분석해 진단했다. 진단과 블록의 대응은 4.6.1절에 적었고, 여기서는 근거 수치를 보고한다. 수치는 `vla_v4_report`가 1단계 실행 파일에서 다시 계산해 `v4_kpi.json`의 `v3_diag` 키에 기록한 값이다. 모두 v3 결과를 본 뒤의 사후 분석이고, 아래의 원인 해석은 검정하지 않은 **가설**이며 v4 후보를 정하는 데만 썼다.

1. **AMR의 정지 작업자 충돌과 저속 편향(K5).** v3 전체 데이터 정책(시드 {{s:k4:v3_diag:amr_full:n_seeds:0}}개)의 실패는 두 양상에 몰려 있었다.
   - 정지 작업자 시나리오에서는 {{s:k4:v3_diag:amr_full:robot_agent_stop:n:0}}회 중 {{s:k4:v3_diag:amr_full:robot_agent_stop:collision_moving:0}}회가 주행 중 충돌이었다. 천천히 다가가다 완전히 멈추지 못하고 밀려 들어가는 양상으로 보였다(시드 0 평균 속도: 정책 {{s:k4:v3_diag:amr_full:robot_agent_stop:policy_speed_seed0:2}}, 전문가 {{s:k4:v3_diag:amr_full:robot_agent_stop:expert_speed:2}} m/s).
   - 빈 통로 시나리오에서는 {{s:k4:v3_diag:amr_full:robot_aisle_free:n:0}}회 중 {{s:k4:v3_diag:amr_full:robot_aisle_free:progress_fail:0}}회가 진행 미달이었다. 시드 0 평균 속도가 {{s:k4:v3_diag:amr_full:robot_aisle_free:policy_speed_seed0:2}} m/s로 전문가 {{s:k4:v3_diag:amr_full:robot_aisle_free:expert_speed:2}} m/s보다 느렸다.
   - 시드 0 정책의 개루프 위험 프레임 MAE는 {{s:k4:v3_diag:amr_open_loop_seed0:mae_hazard}}로 전체 평균 {{s:k4:v3_diag:amr_open_loop_seed0:mae}}의 약 {{s:k4:v3_diag:amr_open_loop_seed0:ratio:1}}배였다. 위험 프레임은 개루프 테스트 프레임의 {{p:k4:v3_diag:amr_open_loop_seed0:hazard_frac}}({{s:k4:v3_diag:amr_open_loop_seed0:n_hazard:0}}/{{s:k4:v3_diag:amr_open_loop_seed0:n_frames:0}})에 불과했다.
2. **위험 시나리오의 추세 단서(K1·K3).** CARE 2%의 성공률이 낮은 칸은 보행자 횡단, 정지-출발, 선행차 급제동이었고, 보행자 횡단은 전체 데이터에서도 낮았다(표 26). 접근 속도·횡이동처럼 1초 단위 추세가 중요한 장면이다. 2프레임 특징이 이 추세를 담지 못한다는 해석은 가설이며, 새 테스트에서 P2만 절제해 확인하지는 않았다.
3. **실영상 점수기와 단순 기준선(K7).** CARE 실영상 점수기의 제동 시작 AUROC는 {{s:k4:v3_diag:comma_scorer_auroc:mean}}(fold별 {{s:k4:v3_diag:comma_scorer_auroc:min:2}}~{{s:k4:v3_diag:comma_scorer_auroc:max:2}})로 정책({{s:k:K7:value}})과 비슷한 우연 수준이었다. 반면 같은 시험 프레임과 onset 정의에서 정책 없이 현재 자차 가속도 $-a_t$ 하나만 점수로 쓰면 {{s:k4:v3_diag:comma_neg_accel_auroc:mean}}였다. 다만 제동 라벨이 가속도 임계로 정의되므로(3.6.4절), 이 대비는 '자차 운동 상태가 라벨과 가깝다'는 것을 보일 뿐 영상 정보의 가치를 직접 재지는 않는다.
4. **반사실 스타일별 저속 편향(K6).** v3 반사실 결과를 스타일별로 나누면, CARE 2% 정책은 '보통'·'민첩' 지시(목표 속도 평균 {{s:k4:v3_diag:cf_care_by_style:lang_normal:v_target:1}} m/s)에서 언어 유무와 관계없이 에피소드 평균 속도가 {{s:k4:v3_diag:cf_care_by_style:lang_normal:mean_speed:1}}~{{s:k4:v3_diag:cf_care_by_style:lang_brisk:mean_speed:1}} m/s(언어 있음), {{s:k4:v3_diag:cf_care_by_style:nolang_normal:mean_speed:1}} m/s(언어 없음)에 머물렀고, 자유주행 속도 오차는 언어 있음 {{s:k4:v3_diag:cf_care_by_style:lang_brisk:speed_error:2}}~{{s:k4:v3_diag:cf_care_by_style:lang_normal:speed_error:2}} m/s, 언어 없음 {{s:k4:v3_diag:cf_care_by_style:nolang_normal:speed_error:2}}~{{s:k4:v3_diag:cf_care_by_style:nolang_brisk:speed_error:2}} m/s였다. 전체 데이터 정책은 언어가 있으면 오차가 절반 가까이 줄었다(표 28). 데이터가 적을 때 지시문의 숫자 → 목표 속도 대응을 배우지 못한 것으로 해석했다(가설).
5. **S1 측정 결과(K4).** 초기 진단은 선택 전체의 위험 비중(표 27)이 CARE와 트리거 혼합에서 비슷하다는 관찰에서 출발했으나, 이는 점수 몫의 위험 비중을 잘못 읽은 것이었다. 점수 몫만 보면 CARE의 위험 비중은 {{s:k4:v3_diag:s1:care_nocap:score_hazard_frac_mean:2}}로, 감속 트리거 혼합({{s:k4:v3_diag:s1:mix_trigger_nocap:score_hazard_frac_min:2}}~{{s:k4:v3_diag:s1:mix_trigger_nocap:score_hazard_frac_max:2}})보다 오히려 높았다. 또 CARE 2%의 점수 몫 36클립은 시드 0·1·2 모두에서 이미 서로 다른 {{s:k4:v3_diag:s1:care_nocap:n_score_episodes_min:0}}개 에피소드에서 나왔다. 따라서 에피소드당 1클립 상한(S1, $c$=1)은 CARE의 선택을 바꾸지 않고 트리거 혼합의 선택만 바꾼다(3.6.2절). S1은 비교 기준선만 바꾸므로 개발 비교 전에 기각했고, v4에는 K4를 직접 겨냥한 개선이 없다.

### 5.4 v4 개발 세트 선택(적응적 선택, 확증 근거 아님)

v4 조합은 새 개발 세트(주행 84 에피소드, AMR 72 에피소드)에서 CARE 2%, 시드 3개로 골랐다. 후보, 채택 규칙, 후보를 더한 경위는 4.6절(표 18)에 있다. 표 31은 후보별 개발 세트 성공률이고, 표 32는 주행 반사실 개발 세트의 속도 오차다.

{{table:v4_dev}}

표 31. 2단계 후보의 개발 세트 성공률(CARE 2%, 시드 0·1·2 평균 ± 시드 간 표준편차). 주행은 84 에피소드, AMR은 72 에피소드다. '채택'은 4.6.3절 규칙으로 고른 조합이고, 주행 'T1 채택'은 T1 규칙의 판정이다. 출처: `experiments/exp_130_vla_v4/cache/v4_choice_{driving,robot}.json`.

{{table:v4_dev_lang}}

표 32. 주행 반사실 개발 세트(42 에피소드)의 자유주행 속도 오차(m/s, 시드 0·1 평균)와 감소율(1 − 언어 있음/언어 없음). 마지막 열은 시드 0 / 시드 1의 '언어 있음·언어 없음' 값이다. 언어 없음 정책에서는 P6·P7 경로도 꺼지므로 '전부', '전부+P6', '전부+P7'의 언어 없음 값은 같다.

**채택.** 주행과 AMR 모두 '전부+P7'(P2+P3+T1+P4+P6+P7)이 채택되었다.

- **주행**: 개발 세트 성공률은 v3 기준 {{s:k4:dev_choice:driving:dev_success:v3}}에서 단일 블록 {{s:k4:dev_choice:driving:dev_success:p3}}~{{s:k4:dev_choice:driving:dev_success:t1}}, '전부' {{s:k4:dev_choice:driving:dev_success:all}}, '전부+P6' {{s:k4:dev_choice:driving:dev_success:all_p6}}, '전부+P7' {{s:k4:dev_choice:driving:dev_success:all_p7}}의 순서로 올랐다(P5 제외, 표 31). 단일 블록과 그것이 겨냥한 지표의 대응은 개발 세트에서도 확인되지 않았다(예: P3 단독 {{s:k4:dev_choice:driving:dev_success:p3}}, AMR P4 단독 {{s:k4:dev_choice:robot:dev_success:p4}}).
- T1 규칙은 충족되었으나(언어 있음 오차 {{s:k4:dev_choice:driving:devcf:t1:lang:2}} ≤ {{s:k4:dev_choice:driving:devcf:v3:lang:2}} m/s), '전부'가 이미 T1을 포함하므로 결과에 영향이 없다.
- P6 규칙: '전부+P6'의 성공률은 '전부' 이상이었고, 언어 있음 오차({{s:k4:dev_choice:driving:devcf:all_p6:lang:2}} m/s)도 '전부'({{s:k4:dev_choice:driving:devcf:all:lang:2}} m/s) 이하였다. 두 조건을 충족했으나, 오차 조건은 시드 2개에서 0.02 m/s 미만의 차이로 충족된 것이다.
- P7 규칙: '전부+P7'(성공률 {{s:k4:dev_choice:driving:dev_success:all_p7}}, 언어 있음 오차 {{s:k4:dev_choice:driving:devcf:all_p7:lang:2}} m/s)이 두 조건을 충족해 최종 채택되었다.
- **반사실 개발 세트의 K6 예측(표 32)**: '전부'와 '전부+P6'에서는 언어를 넣은 정책의 오차가 언어를 뺀 정책({{s:k4:dev_choice:driving:devcf:all:nolang:2}} m/s)보다 컸다. '전부+P7'에서 처음으로 언어 있음이 언어 없음보다 작아졌으나, 감소율은 {{p:k4:dev_choice:driving:devcf:all_p7:reduction}}로 K6 목표(30%)에 못 미쳤다. 시드별로도 시드 0에서는 언어 있음({{s:k4:dev_choice:driving:devcf:all_p7:by_seed_lang:0:2}} m/s)이 언어 없음({{s:k4:dev_choice:driving:devcf:all_p7:by_seed_nolang:0:2}} m/s)보다 컸다. 즉 개발 세트는 채택 조합의 K6 미달을 예측했다.
- **AMR**: 단일 블록과 '전부'는 v3 기준({{s:k4:dev_choice:robot:dev_success:v3}})과 비슷했고('전부' {{s:k4:dev_choice:robot:dev_success:all}}), '전부+P6' {{s:k4:dev_choice:robot:dev_success:all_p6}}을 거쳐 '전부+P7'에서 크게 올랐다({{s:k4:dev_choice:robot:dev_success:all_p7}}). AMR에는 반사실 조건을 적용하지 않는다.
- **전체 데이터(시드 0, 참고).** 주행: v3 기준 {{s:k4:dev_choice:driving:dev_full_success:v3}}, '전부' {{s:k4:dev_choice:driving:dev_full_success:all}} / AMR: v3 기준 {{s:k4:dev_choice:robot:dev_full_success:v3}}, '전부' {{s:k4:dev_choice:robot:dev_full_success:all}}.
- **최종 조합.** 두 도메인 모두 '전부+P7'을 채택했다. 두 조합은 P7 이득 $k$만 다르다(주행 3.75, AMR 1.5).

**탈락·기각.**

- **S1(점수 몫 에피소드 상한)**: 개발 비교 전에 측정으로 기각했다(5.3절 5).
- **P5(자차 운동 이력)**: 기본 규칙(최댓값 선택)에서 선택되지 않아 탈락했다. v3 정책에 P5만 더한 후보의 주행 개발 세트 성공률({{s:k4:dev_choice:driving:dev_success:p5}})은 v3 기준보다도 낮았고 시드 간 편차가 컸다(표 31). 자차의 최근 속도 변화를 따라 하는 관성 추종이 원인일 수 있으나 확인하지 않았다. AMR에서는 v3 기준과 비슷했다({{s:k4:dev_choice:robot:dev_success:p5}}). 이 탈락 때문에 v4 실영상 정책에도 속도 이력이 없다(5.5절 (5)).

**해석상 주의.** 이 선택은 확증 근거가 아니다. 개발 세트는 여러 번 재사용되었고, 후보 집합 자체가 앞선 결과를 보고 늘어났다(P5·P6·P7). 최대 성공률 후보를 고르는 규칙은 개발 세트 값을 낙관적으로 만든다. 실제로 '전부+P7'의 주행 개발 세트 값({{s:k4:dev_choice:driving:dev_success:all_p7}})은 새 테스트 값({{s:k4:K1:value}}, 5.5절)보다 조금 높았다. 블록별 기여도 개발 세트의 시드 3개 비교일 뿐이며, 새 테스트에서 블록별 절제는 하지 않았다. AMR 개발 세트에서 P7을 더할 때 크게 오른 것도 시드 3개 결과이며, AMR 새 테스트에서 P7 몫을 분리하지 않았다. 확증되는 것은 '채택된 v4 설정 전체'의 효과뿐이다(5.5절의 H-v).

### 5.5 2차 확증(v4, 새 테스트 세트)

2차 확증은 사전 등록(docs/37, 커밋 `6f97127`, 2026-10-04 23:26:29 UTC)을 커밋한 뒤, 처음 보는 새 테스트 세트(주행 1000000번대 147 에피소드, 반사실 1050000번대, AMR 1100000번대)에서 수행했다. 풀·점수기·풀 점수·선별 하이퍼파라미터는 v3와 같고, 정책·학습 설정만 5.4절의 채택 조합이다. 같은 새 테스트에서 v3 설정(CARE 2%·무작위 2%, 시드 10개)도 함께 평가해 개선 효과를 추정했다. 실영상에서는 아무것도 튜닝하지 않고 주행 조합을 그대로 적용했다. 새 세트의 전문가 성공률은 주행 {{s:k4:expert_test_success:driving}}, 위험 시나리오 {{s:k4:expert_test_success:hazard}}, AMR {{s:k4:expert_test_success:robot}}다(4.7.1절).

**(1) 주행 폐루프(K1~K3).** 표 33은 새 테스트의 주행 결과이고, 표 34는 시나리오별 성공률이다.

{{table:v4_driving}}

표 33. 새 테스트 세트의 주행 결과(평균 ± 시드 표준편차; 위험 시나리오 성공률도 시드 표준편차를 함께 적음). '설정' 열의 v3는 v3 정책·학습 설정, v4는 채택 조합이다. 선별 방법·풀·점수기는 두 설정이 같다. '(탐색)'이 붙은 행(P3·P4를 뺀 v4, v3 설정의 '저장소 + 감속 트리거')은 3차 심사 뒤의 탐색적 절제(새 테스트 재사용, 4.8절·5.6절)이며 사전 등록 확증에 속하지 않는다.

- **K1(점추정 달성)**: v4 CARE 2%의 성공률은 {{s:k4:K1:value}} ± {{s:k4:K1:sd}}(시드 {{s:k4:K1:n_seeds:0}}개)로 목표 0.80을 넘었다. 시드·에피소드 부트스트랩 95% 구간은 [{{u:k4:K1:lo}}, {{u:k4:K1:hi}}]이고, 재표집의 {{p:k4:K1:p_ge_goal}}에서 목표 이상이었다.
- **K1·K3는 선별법을 판별하지 못했다.** 같은 v4 정책에서 무작위 2%({{s:k4:driving:v4:random_shared:success}})도 K1 목표를 넘었고, 저장소 + 감속 트리거({{s:k4:driving:v4:mix_trigger:success}}, 위험 시나리오 {{s:k4:driving:v4:mix_trigger:hazard}})는 K1·K3를 모두 넘었다. K1·K3 달성은 정책 설정 변경의 성과이며, CARE 선별의 고유 효과로 볼 근거는 없다.
- **K2(미달)**: 전체 데이터 성공률({{s:k4:K2:full}})에 대한 비율은 {{s:k4:K2:value}}(부트스트랩 95% 구간 [{{u:k4:K2:lo}}, {{u:k4:K2:hi}}], 목표 이상 비율 {{p:k4:K2:p_ge_goal}})로 목표 0.90에 미달했다. v3 테스트의 값({{s:k:K2:value}})과는 평가 세트가 달라(교차 세트) 개선 폭으로 읽지 않는다.
- **K3(경계 달성)**: 위험 시나리오 성공률은 {{s:k4:K3:value}} ± {{s:k4:K3:sd}}(시드 표준편차)로 목표 0.75를 {{u:k4:K3:margin}} 넘었다. 이는 위험 시나리오 평가 {{s:k4:K3:n_episodes:0}}회(84 에피소드 × 시드 10) 가운데 성공 {{s:k4:K3:margin_episodes:0}}회 분량의 차이다. 시드·에피소드 부트스트랩 95% 구간은 [{{u:k4:K3:lo}}, {{u:k4:K3:hi}}]로 목표를 넓게 포함하며, 재표집에서 목표 이상이 나온 비율은 {{p:k4:K3:p_ge_goal}}에 그쳤다. 사전 등록 판정은 점추정 기준이므로 '달성'으로 보고하지만, 이 불확실성은 반복 실험에서 미달로 바뀔 수 있음을 뜻한다.
- **시나리오별(표 34)**: v4 CARE 2%의 보행자 횡단 성공률은 {{s:k4:driving_scenario:v4:care:vru_crossing}}로, 전문가({{s:k4:driving_scenario:expert:vru_crossing}})와 v4 전체 데이터({{s:k4:driving_scenario:v4:full:vru_crossing}})에 크게 못 미쳤다. P2가 겨냥한 보행자 횡단은 2% 데이터에서 여전히 절반 가까이 실패했다. 선행차 급제동({{s:k4:driving_scenario:v4:care:lead_brake}})과 정체({{s:k4:driving_scenario:v4:care:stop_and_go}})도 전문가(각각 {{s:k4:driving_scenario:expert:lead_brake}}, {{s:k4:driving_scenario:expert:stop_and_go}})와 차이가 컸다.
- 같은 새 테스트에서 v3 설정의 CARE 2% 성공률({{s:k4:driving:v3:care:success}})은 v3 자체 테스트의 값({{s:k:K1:value}})보다 낮았다. 새 테스트 세트가 v3 테스트보다 어려웠을 가능성이 있으며, v3·v4의 절대값 비교는 같은 세트 안에서만 해석한다.

{{table:v4_driving_scenario}}

표 34. 새 테스트 세트의 시나리오별 성공률(예산 2%는 시드 10개, 전체는 시드 5개 평균; 전문가는 자기 자신 기준의 무충돌률). ʰ는 위험 시나리오다. 칸별 신뢰구간은 계산하지 않았으며 탐색적 기술 통계다. v3 행은 같은 새 테스트에서 v3 설정 CARE 2%의 값이다.

**(2) 가설 검정(K4, H-a, H-v).** 표 35는 2단계 사전 등록 가설 검정과 보조 비교다.

| 비교(예산 2%, 시드 10개 대응) | 차이 | 계층 부트스트랩 95% CI | 시드 대응 $t$ 95% CI | 단측 $p$ | Holm 보정 $p$ | 판정 |
|---|---|---|---|---|---|---|
| K4: v4 CARE − v4 저장소 + 감속 트리거 | {{s:k4:K4:diff}} | [{{s:k4:K4:lo}}, {{s:k4:K4:hi}}] | [{{s:k4:K4:seed_t:lo}}, {{s:k4:K4:seed_t:hi}}] | {{s:k4:K4:p_le0:3}} | {{s:k4:K4:p_holm:3}} | 미달 |
| H-a: v4 CARE − v4 무작위(공유 저장소) | {{s:k4:comparisons:care_vs_random_shared:diff}} | [{{s:k4:comparisons:care_vs_random_shared:lo}}, {{s:k4:comparisons:care_vs_random_shared:hi}}] | [{{s:k4:comparisons:care_vs_random_shared:seed_t:lo}}, {{s:k4:comparisons:care_vs_random_shared:seed_t:hi}}] | {{s:k4:comparisons:care_vs_random_shared:p_le0:4}} | {{s:k4:comparisons:care_vs_random_shared:p_holm:3}} | 지지 |
| H-v: v4 CARE − v3 CARE | {{s:k4:comparisons:v4_vs_v3_care:diff}} | [{{s:k4:comparisons:v4_vs_v3_care:lo}}, {{s:k4:comparisons:v4_vs_v3_care:hi}}] | [{{s:k4:comparisons:v4_vs_v3_care:seed_t:lo}}, {{s:k4:comparisons:v4_vs_v3_care:seed_t:hi}}] | < 10⁻⁴ | < 0.001 | 지지 |
| (탐색) v4 무작위 − v3 무작위 | {{s:k4:comparisons:v4_vs_v3_random_shared:diff}} | [{{s:k4:comparisons:v4_vs_v3_random_shared:lo}}, {{s:k4:comparisons:v4_vs_v3_random_shared:hi}}] | [{{s:k4:comparisons:v4_vs_v3_random_shared:seed_t:lo}}, {{s:k4:comparisons:v4_vs_v3_random_shared:seed_t:hi}}] | < 10⁻⁴ | 해당 없음 | - |
| (탐색) v3 CARE − v3 무작위(새 테스트) | {{s:k4:comparisons:v3_care_vs_random_shared:diff}} | [{{s:k4:comparisons:v3_care_vs_random_shared:lo}}, {{s:k4:comparisons:v3_care_vs_random_shared:hi}}] | [{{s:k4:comparisons:v3_care_vs_random_shared:seed_t:lo}}, {{s:k4:comparisons:v3_care_vs_random_shared:seed_t:hi}}] | {{s:k4:comparisons:v3_care_vs_random_shared:p_le0:4}} | 해당 없음 | - |
| (탐색) v4 저장소 + 감속 트리거 − v4 무작위 | {{s:k4:explore:k4:v4_mix_trigger_vs_random_shared:diff}} | [{{s:k4:explore:k4:v4_mix_trigger_vs_random_shared:lo}}, {{s:k4:explore:k4:v4_mix_trigger_vs_random_shared:hi}}] | - | {{s:k4:explore:k4:v4_mix_trigger_vs_random_shared:p_le0:3}} | 해당 없음 | - |

표 35. v4 사전 등록 가설 검정(docs/37 3절, 새 테스트 세트). Holm 보정 군은 K4·H-a·H-v다. H-v와 v4 무작위 − v3 무작위는 계층 부트스트랩 10,000회 가운데 차이 ≤ 0인 재표집이 하나도 없었으므로 $p$ < 10⁻⁴(해상도 한계)로 적었다. '(탐색)' 행은 보정하지 않은 보조 비교다. 마지막 행은 사전 등록 확증 실행의 자료로 계산했으나 사전 등록한 비교가 아니다.

- **K4(미달)**: v4 CARE와 v4 트리거 혼합의 차이는 {{s:k4:K4:diff}} [95% CI {{s:k4:K4:lo}}, {{s:k4:K4:hi}}]이고, 구간이 0을 포함했다(Holm 보정 $p$={{s:k4:K4:p_holm:3}}). 두 방법의 성공률(각각 {{s:k4:driving:v4:care:success}}, {{s:k4:driving:v4:mix_trigger:success}})은 거의 같았다. v3(5.2절)와 달리 이번에는 시드 대응 $t$ 구간도 0을 넓게 포함했다. 단, v4에서는 P3·P4가 모든 방법의 학습에 점수기 출력을 넣으므로, 이 K4는 점수기 정보가 학습 신호로 이미 모든 방법에 주어진 상태의 **선택 역할만의 차이**이며 v3의 K4와 같은 양이 아니다(3.7절). 동등성(효과 없음)은 2단계에서 사전 등록하지 않았으므로 주장하지 않는다. 3단계에서 점수기 출력을 학습에 쓰지 않는 조건으로 사전 등록한 동등성 검정(H-K4e)도 동등을 보이지 못했다(5.7절 (2)).
- **H-a(지지)**: 공유 저장소를 고정한 무작위 대비 우위는 v4에서도 유지되었다({{s:k4:comparisons:care_vs_random_shared:diff}}, Holm 보정 $p$={{s:k4:comparisons:care_vs_random_shared:p_holm:3}}). 같은 정책에서 트리거 혼합 − 무작위도 {{s:k4:explore:k4:v4_mix_trigger_vs_random_shared:diff}} [95% CI {{s:k4:explore:k4:v4_mix_trigger_vs_random_shared:lo}}, {{s:k4:explore:k4:v4_mix_trigger_vs_random_shared:hi}}](탐색)로 0을 포함하지 않았다. 그러나 이 v4 비교들은 모두 점수기 출력으로 학습한 정책의 것이다. 점수기 출력 없이 학습한 v3에서는 CARE − 무작위(H-a)가 1단계에서 지지되었으나({{s:k:comparisons:care_vs_random_shared_b0.02:diff}}, 표 25), 트리거 혼합 − 무작위는 0을 포함했다({{s:k:comparisons:mix_trigger_vs_random_shared_b0.02:diff}} [95% CI {{s:k:comparisons:mix_trigger_vs_random_shared_b0.02:lo}}, {{s:k:comparisons:mix_trigger_vs_random_shared_b0.02:hi}}], 표 25; 2단계 새 테스트의 v3 설정도 같음, 5.6.2절).
- **H-v(지지)**: 같은 새 테스트에서 v4 설정은 v3 설정보다 CARE 2% 성공률이 {{s:k4:comparisons:v4_vs_v3_care:diff}} 높았다. 무작위 2%에서도 개선 폭이 비슷했다({{s:k4:comparisons:v4_vs_v3_random_shared:diff}}, 탐색). 정책·학습 블록의 개선은 선별 방법과 거의 무관하게 더해졌다. 2단계에서는 어느 블록이 기여했는지 분리되지 않았으며, 해석적 제어기(P7)가 성공률에도 작용했을 수 있었다(3.6.2절). 3단계에서는 개선 효과가 세 번째 테스트에서 재현되었고(H-v3), 블록별 제거로 P2·T1의 기여는 검출했으나 P7의 성공률 기여(H-P7)는 지지되지 않았다(5.7절 (2)·(3)).
- **v3 효과의 재현(탐색)**: 같은 새 테스트에서 v3 설정의 CARE − 무작위({{s:k4:comparisons:v3_care_vs_random_shared:diff}})는 v3 테스트의 H-a({{s:k:comparisons:care_vs_random_shared_b0.02:diff}})와 방향·크기가 비슷했다. 다만 계층 CI 하한이 0에 걸쳤고({{s:k4:comparisons:v3_care_vs_random_shared:lo}}), 시드 대응 $t$ 구간은 0을 포함하지 않았다.
- **충돌률**: 낮은 순서는 트리거 혼합({{s:k4:driving:v4:mix_trigger:collision}}) < CARE({{s:k4:driving:v4:care:collision}}) < 무작위({{s:k4:driving:v4:random_shared:collision}})였다(표 33). 트리거 혼합은 위험 시나리오 성공률({{s:k4:driving:v4:mix_trigger:hazard}})도 CARE({{s:k4:driving:v4:care:hazard}})보다 높았다. 모두 검정하지 않은 기술 통계다.

**(3) 언어 지시 준수(K6).** 표 36은 새 반사실 세트의 언어 평가다.

{{table:v4_language_cf}}

표 36. 새 반사실 세트(시드 1050000번대, 63 에피소드)의 언어 평가(v4 설정, 시드 5개). 언어를 빼면 지시문에서 목표 속도를 읽는 P6·P7 경로도 함께 꺼진다. '자유주행 프레임'은 속도 오차를 계산한 프레임 수의 에피소드 평균으로, 정책의 궤적에 따라 달라진다(3.7절).

- **K6(점추정 달성)**: v4 CARE 2%에서 언어를 넣으면 속도 추종 오차가 {{s:k4:K6:nolang:speed_error:2}} → {{s:k4:K6:lang:speed_error:2}} m/s로 {{p:k4:K6:speed_error_reduction}} 줄었다(목표 30%). 전체 데이터(보조)에서는 {{s:k4:K6:by_data:full:nolang:speed_error:2}} → {{s:k4:K6:by_data:full:lang:speed_error:2}} m/s로 {{p:k4:K6:by_data:full:speed_error_reduction}} 줄었다.
- **시드별 감소율.** CARE 2%의 시드별 감소율은 {{p:k4:K6:by_seed:0:1}}, {{p:k4:K6:by_seed:1:1}}, {{p:k4:K6:by_seed:2:1}}, {{p:k4:K6:by_seed:3:1}}, {{p:k4:K6:by_seed:4:1}}(시드 0~4)였다. 다섯 시드 중 {{s:k4:K6:n_seeds_reduced:0}}개에서만 오차가 줄었고, 두 시드에서는 줄지 않았다. 시드 간 표준편차는 {{s:k4:K6:sd_seed_reduction:2}}이고, 시드 부트스트랩 95% 구간은 [{{p:k4:K6:lo}}, {{p:k4:K6:hi}}]이다. 반사실 에피소드의 변동까지 넣은 시드·에피소드 계층 구간은 [{{p:s4:k6_hier_stage2:lo}}, {{p:s4:k6_hier_stage2:hi}}](목표 이상 비율 {{p:s4:k6_hier_stage2:p_ge_goal:0}}, 보조)로, 하한이 0 아래였다.
- **시드 간 차이의 정체(4차 심사 대응).** 같은 학습 시드의 같은 정책을 세 번째 반사실 세트에서 평가하자 시드별 감소율이 크게 바뀌었다(시드 0: {{p:k4:K6:by_seed:0:1}} → {{p:k3:k6:K6:by_seed:0:1}}, 시드 2: {{p:k4:K6:by_seed:2:1}} → {{p:k3:k6:K6:by_seed:2:1}}, 5.7절 (4)). 따라서 2단계에서 본 시드 간 차이('쌍봉')는 학습 불안정이 아니라 주로 63 에피소드 평가의 변동이었다. 이전 원고의 '학습 잔차가 제어기를 상쇄하는 시드가 있다'는 해석은 철회한다. 전체 데이터에서는 다섯 시드가 모두 비슷하게 줄었다([{{p:k4:K6:by_data:full:lo}}, {{p:k4:K6:by_data:full:hi}}]). 개발 세트에서 같은 조합의 감소율은 {{p:k4:dev_choice:driving:devcf:all_p7:reduction}}로 목표에 못 미쳤다(5.4절).
- 성공률과 자유주행 프레임 수는 언어 있음/없음에서 비슷했다(CARE {{s:k4:K6:lang:success}} / {{s:k4:K6:nolang:success}}, 프레임 {{s:k4:K6:lang:free_frames:0}} / {{s:k4:K6:nolang:free_frames:0}}). 프레임 집합이 조건마다 다르다는 선택 효과는 남는다.
- 신중−민첩 추종 간격 차는 언어를 넣으면 CARE {{s:k4:K6:nolang:style_sep:2}} → {{s:k4:K6:lang:style_sep:2}} s, 전체 데이터 {{s:k4:K6:by_data:full:nolang:style_sep:2}} → {{s:k4:K6:by_data:full:lang:style_sep:2}} s로 커졌다.
- **해석.** v4에는 P7(지시문의 목표 속도로 만든 비례 제어 기준 $a^0$ + 학습 잔차)이 들어 있고, 언어를 빼면 P6·P7도 꺼진다. 따라서 이 감소는 '학습된 언어 조건화'와 '지시문 숫자의 결정적 해석 + 해석적 제어기'를 합친 효과다. 3차 심사 뒤의 탐색적 절제(5.6.1절)에서 이 둘을 나누어 보았고, 그 결과 K6 달성의 메커니즘은 P6·P7 경로였다. 사전 등록 판정('달성')은 바꾸지 않고, 해석은 '조건부(메커니즘: P6·P7)'로 둔다(6.5절, 표 52). 이 해석은 세 번째 테스트에서 확증 가설 H-L로 다시 검정했으며, H-L은 지지되지 않아 사전 등록 해석 규칙대로 같은 결론으로 서술한다(5.7절 (4)). 이는 규칙에 따른 서술이며 학습 언어 경로의 기여가 없음을 확인한 것은 아니다.

**(4) AMR(K5).** 표 37은 새 AMR 테스트 결과다.

{{table:v4_robot}}

표 37. 새 AMR 테스트 세트(시드 1100000번대)의 결과(2%는 시드 5개, 전체는 시드 3개). 선별 하이퍼파라미터는 v3 AMR 튜닝값($\lambda$=0, $\rho$=0.95)이다.

- **K5(미달)**: v4 CARE − v4 무작위는 {{s:k4:K5:diff}} [95% CI {{s:k4:K5:lo}}, {{s:k4:K5:hi}}]였다. CI 하한이 비열등 한계 −0.03보다 낮아 비열등을 판정하지 못했다. 시드별 차이의 표준편차는 {{s:k4:K5:sd_seed_diff:3}}, CI 반폭은 {{s:k4:K5:halfwidth:3}}로, 비열등을 판정하기에 구간이 넓었다(6.3절).
- 같은 새 AMR 테스트의 v3 대조는 2단계에 없다(3단계에서 세 번째 AMR 테스트에 두었다, 5.7절 (5)). v3 테스트의 값(CARE {{s:k:K5:care}}, 무작위 {{s:k:K5:random_shared}}, 전체 데이터 {{s:k:robot_full_success}})과 이번 값(CARE {{s:k4:K5:care}}, 무작위 {{s:k4:K5:random_shared}}, 전체 데이터 {{s:k4:robot_full_success}})은 서로 다른 테스트 세트의 값이어서 대응 비교가 아니다. 주행에서는 같은 새 테스트에서 v3 설정이 v3 테스트보다 낮게 나왔으므로(5.5절 (1)), AMR에서도 세트 난이도 차이가 클 수 있다. 따라서 v3 → v4의 AMR 향상은 주장하지 않는다.
- 트리거 혼합({{s:k4:robot:v4:mix_trigger:success}})과 무작위의 성공률은 같았다(표 37).

**(5) 실영상(K7).** 표 38은 v4 설정의 실영상 개루프 결과다.

{{table:v4_comma}}

표 38. v4 설정의 comma.ai 실영상 개루프 결과(v3와 같은 5-fold × 시드 2개, ± 는 10회 실행의 표준편차). 실영상에서는 아무것도 튜닝하지 않고 주행 개발 조합(P7 이득은 주행 값)을 그대로 적용했다. 1단계와 같은 fold를 재사용했으므로 새 데이터의 확증이 아니다. GT 위험 라벨이 없어 오라클은 없고, 감속 트리거 단독은 v4에서 평가하지 않았다.

- **K7(미달)**: v4 전체 데이터 정책의 제동 시작 AUROC는 {{s:k4:K7:value}} ± {{s:k4:K7:sd}}에 그쳤다(v3 {{s:k:K7:value}}, 같은 fold). fold·시드별 값은 {{s:k4:K7:min:2}}~{{s:k4:K7:max:2}} 범위에 흩어졌다(표 38).
- 같은 시험 프레임에서 정책 없이 현재 자차 가속도 $-a_t$만 쓴 기준선의 AUROC는 {{s:k4:K7:baseline_neg_accel}}였다. 제동 라벨이 가속도 임계($a<-0.6$ m/s²)로 정의되므로, 이 기준선은 '가속도의 연속성'을 재며 라벨 정의에 가까운 값이다. 정책은 $a_t$를 입력으로 받지 않는다. 따라서 정책과 기준선의 차이는 영상 이해의 부족만이 아니라 입력 정보의 차이도 반영한다. P5가 개발 세트에서 탈락해(5.4절) v4 정책에도 속도 이력 입력이 없다. P5를 넣은 정책의 실영상 결과는 3단계에서 탐색적으로 보고한다(5.7절 (6)).
- 예산 10·20%의 방법 간 AUROC는 0.52~0.57 범위로, 방법 간 차이를 가릴 분해능이 없었다.

### 5.6 3차 심사 후 탐색적 절제(새 테스트 재사용)

이 절의 결과는 모두 **탐색적(새 테스트 재사용)**이다. 2단계 새 테스트 결과를 본 뒤 같은 세트에서 실행했으므로 확증이 아니며, 사전 등록 판정을 바꾸지 않는다. 설계는 4.8절(표 20)에 있다. 이 절의 역할은 3단계에서 확증할 가설을 세운 데 있고, 그 가설의 확증 결과는 5.7절이 대신한다. 따라서 여기서는 무엇을 보고 어떤 가설을 세웠는지만 적는다. 이 절의 정책 가운데 'P3·P4 제외 v4'(시드 0~9)와 'P6·P7만'·'전부+P6'(시드 0~4)은 3단계에서도 같은 학습 시드의 같은 정책으로 다시 쓰였다(4.9.1절).

#### 5.6.1 K6 분해

표 39는 v4의 K6을 학습 언어 경로, 결정적 수치 목표(P6), 해석적 제어기(P7)로 나누어 본 결과다.

{{table:v4_explore_k6}}

표 39. K6 분해(탐색적, 새 테스트 재사용; CARE 2%, 반사실 새 세트 63 에피소드, 시드 0~4). 감소율은 'v4 전부+P7, 언어 없음'({{s:k4:explore:k6_alias:full_nolang:speed_error:2}} m/s)에 대한 비율이며, 대괄호는 같은 시드끼리 대응시킨 시드 부트스트랩 95% 구간, 그 뒤는 오차가 줄어든 시드 수다. 제어기 단독은 결정적이라 실행이 1회다. 성공률은 반사실 세트의 식 (K0) 성공률이다.

- **관찰.** P7 없이 P6만으로는 언어 효과가 없었다(감소율 {{p:k4:explore:k6_alias:p6_lang:reduction_vs_nolang}}). 학습 언어 경로 없이 P6·P7만 쓴 정책의 오차({{s:k4:explore:k6_alias:p67_only:speed_error:2}} m/s)가 가장 작았고, v4(학습 언어 경로 + P6·P7, {{s:k4:explore:k6_alias:full_lang:speed_error:2}} m/s)와의 시드 대응 차이는 {{s:k4:explore:k6_learned_lang_path:diff:2}} m/s [{{s:k4:explore:k6_learned_lang_path:lo:2}}, {{s:k4:explore:k6_learned_lang_path:hi:2}}]로 판별되지 않았다. 학습 없이 제어기 $a^0$만 쓰면 오차가 {{s:k4:explore:k6_alias:ctrl:speed_error:2}} m/s, 성공률이 {{s:k4:explore:k6_alias:ctrl:success}}에 그쳐 학습 정책 + 잔차보다 나빴다.
- **세운 가설.** 'K6 달성은 지시문 숫자를 결정적으로 읽는 P6·P7 경로 덕분이며, 학습 언어 경로는 오차를 더 줄이지 않는다'(H-L로 3단계에 등록). 'P7은 폐루프 성공률에도 기여한다'(H-P7). 스타일별 추종 간격 분리(표 36)는 P6·P7이 다루지 않는 정보여서 FiLM 경로가 배운 것일 수 있으나 이 절제에서 따로 재지 않았다.
- **3단계 확증 결과(5.7절 (4)).** H-L은 지지되지 않았고 'P6·P7만'의 오차가 v4 언어 있음 이하였으므로, 사전 등록 해석 규칙에 따라 위 해석으로 서술한다(학습 언어 경로의 기여가 없음을 확인한 것은 아니다). 'P6만으로는 언어 효과 없음'은 세 번째 세트에서 작은 감소({{p:k3:k6:p6_reduction}})로 나와 같은 방향으로 재현되지 않았다(보조).

#### 5.6.2 K4 정화와 원인 분리

표 40은 점수기 출력을 학습 신호로 쓰지 않은 v4와, 같은 새 테스트의 v3 설정 트리거 혼합을 포함한 비교다.

{{table:v4_explore_k4}}

표 40. 점수기 출력을 학습에서 뺀 비교와 v3 설정의 트리거 혼합(탐색적, 새 테스트 재사용; 주행 새 테스트, 예산 2%, 시드 0~9 대응). 'P3·P4 제외 v4'는 v4 조합에서 점수기 출력을 쓰는 보조 헤드(P3)와 위험 가중 손실(P4)을 뺀 설정(P2+T1+P6+P7)이다. CI는 시드·에피소드 계층 부트스트랩 95% 구간이고, 단측 $p$는 보정하지 않았다. 'v4: 트리거 혼합 − 무작위' 행은 사전 등록 확증 실행의 자료로 계산한 값으로, 표 35 마지막 행과 같다.

- **관찰.** CARE − 트리거 혼합은 P3·P4 제외 v4({{s:k4:explore:k4:noaux_care_vs_mix_trigger:diff}})와 새 테스트의 v3 설정({{s:k4:explore:k4:v3_care_vs_mix_trigger:diff}})에서 모두 0을 포함했다. P3·P4 제외 v4의 CARE − 무작위({{s:k4:explore:k4:noaux_care_vs_random_shared:diff}} [{{s:k4:explore:k4:noaux_care_vs_random_shared:lo}}, {{s:k4:explore:k4:noaux_care_vs_random_shared:hi}}])도 0을 포함해, v4 H-a({{s:k4:comparisons:care_vs_random_shared:diff}})보다 점추정이 작았다. 다만 두 선별 효과의 차이는 2단계에서 검정하지 않았고, 같은 자료로 사후 계산한 차이({{s:s4:ha_diff_stage2_explore:diff}} [{{s:s4:ha_diff_stage2_explore:lo}}, {{s:s4:ha_diff_stage2_explore:hi}}], 보조)는 0을 넓게 포함했다. 점수기 출력 없이 학습한 v3에서는 1단계 H-a가 지지되었다(표 25).
- **관찰(혼합 구조).** 트리거 혼합 − 무작위는 v4에서만 0을 포함하지 않았고({{s:k4:explore:k4:v4_mix_trigger_vs_random_shared:diff}}), 새 테스트의 v3 설정({{s:k4:explore:k4:v3_mix_trigger_vs_random_shared:diff}})과 P3·P4 제외 v4에서는 0을 포함했다. 같은 새 테스트에서 v3 설정의 CARE − 트리거 혼합도 0 근처였으므로, v3 → v4의 K4 축소를 '정책이 강해졌기 때문'으로 설명할 근거는 없다(평균 회귀와 구별되지 않음).
- **세운 가설.** 'v4 정책에서 점수기 출력을 학습에서 빼도 CARE 몫이 무작위 몫보다 낫다'(H-a3n)와, 그 조건의 CARE와 트리거 혼합의 동등성(H-K4e)을 3단계에 등록했다.
- **3단계 확증 결과(5.7절 (2)).** H-a3n은 지지되지 않았고 H-K4e는 동등을 보이지 못했다. 같은 조건의 트리거 혼합 − 무작위는 {{s:k3:tests:mix_vs_random_noaux:diff}} [{{s:k3:tests:mix_vs_random_noaux:lo}}, {{s:k3:tests:mix_vs_random_noaux:hi}}], v4의 트리거 혼합 − 무작위도 이번에는 {{s:k3:tests:mix_vs_random_v4:diff}} [{{s:k3:tests:mix_vs_random_v4:lo}}, {{s:k3:tests:mix_vs_random_v4:hi}}]로 0을 포함했다(보조). 혼합 구조 자체의 이득은 세 번째 테스트에서도 확인되지 않았다.

### 5.7 3차 확증(세 번째 테스트)

3차 확증은 사전 등록(docs/39, 커밋 `e0ea4a5`, 2026-10-06 16:09:21 UTC)을 커밋한 뒤, 처음 보는 세 번째 테스트 세트(주행 1200000번대 147 에피소드, 반사실 1250000번대 63 에피소드, AMR 1300000번대 72 에피소드)에서 수행했다. 풀·점수기·선별·학습과 v4 정책 설정은 2단계와 같다. 설계와 가설은 4.9절(표 21·22)에 있다. 이 절의 결과는 실영상 P5((6))를 빼면 모두 사전 등록 확증이다. 단, 3단계 정책의 대부분은 2단계와 같은 학습 시드의 같은 정책이므로(표 21 마지막 열, 4.9.1절), 이 확증은 같은 학습 정책을 새 평가 에피소드에서 다시 검정한 것이고 학습 시드 표집에 대한 독립 반복이 아니다. 새 세트의 전문가 성공률은 주행 {{s:k3:expert_success}}(위험 시나리오 {{s:s4:expert:stage3:hazard}}, 보행자 횡단 {{s:s4:expert:stage3:vru_crossing}}), AMR(주행 중 무충돌) {{s:k3:robot:expert_success}}이다.

**(1) KPI 재현(주행 K1~K3).** 표 41은 세 번째 테스트의 주행 결과다.

{{table:third_driving}}

표 41. 세 번째 테스트 세트(시드 1200000번대, 147 에피소드)의 주행 결과(평균 ± 시드 표준편차). 'v4(전부+P7)'은 2단계 채택 조합이고, 'P3·P4 제외'는 점수기 출력을 학습에 쓰지 않는 v4, '전부+P6'은 P7을 뺀 조합, 'P2 제외' 등은 v4에서 그 블록만 뺀 조합(블록별 제거, 시드 5개)이다. 위험 시나리오 성공률은 시드 평균이다. 선별·풀·점수기는 모든 행이 같다.

- **K1(달성)**: v4 CARE 2%의 성공률은 {{s:k3:kpi:K1:value}}(시드 {{s:k3:kpi:K1:n_seeds:0}}개), 시드·에피소드 부트스트랩 95% 구간 [{{u:k3:kpi:K1:lo}}, {{u:k3:kpi:K1:hi}}], 재표집에서 목표 이상 비율 {{p:k3:kpi:K1:p_ge_goal}}였다.
- **K2(달성)**: 전체 데이터({{s:k3:kpi:K2:full}})에 대한 비율은 {{s:k3:kpi:K2:value}} [{{u:k3:kpi:K2:lo}}, {{u:k3:kpi:K2:hi}}]로 점추정이 목표 0.90을 넘었다. 다만 목표 이상 비율은 {{p:k3:kpi:K2:p_ge_goal}}로, 구간이 목표를 넓게 포함한다.
- **K3(달성)**: 위험 시나리오 성공률은 {{s:k3:kpi:K3:value}} ± {{s:k3:kpi:K3:sd}}(시드 표준편차), 95% 구간 [{{u:k3:kpi:K3:lo}}, {{u:k3:kpi:K3:hi}}], 목표 이상 비율 {{p:k3:kpi:K3:p_ge_goal}}였다. 2차의 경계 달성({{s:k4:K3:value}})보다 여유가 컸다.
- **K1·K3는 여전히 선별법을 판별하지 않는다.** 같은 v4 정책에서 무작위 2%({{s:k3:tests:H-a3:b}})와 트리거 혼합({{s:k3:tests:K4:b}})도 K1 목표를 넘었고, 두 방법의 위험 시나리오 성공률도 K3 목표를 넘었다(표 41).
- **시나리오별(표 42, 보조).** v4 CARE 2%에서 가장 낮은 시나리오는 두 세트 모두 보행자 횡단이었다(2차 {{s:s4:scenario:stage2:v4:care:vru_crossing}}, 3차 {{s:s4:scenario:stage3:v4:care:vru_crossing}}; 전문가는 3차에서 {{s:s4:scenario:stage3:expert:vru_crossing}}). 3차의 정체({{s:s4:scenario:stage3:v4:care:stop_and_go}})와 선행차 급제동({{s:s4:scenario:stage3:v4:care:lead_brake}})도 전문가와 거리가 남았다.

{{table:supp4_scenario_23}}

표 42. 2·3단계 시나리오별 성공률(예산 2%는 시드 10개, 전체 100%는 시드 5개의 평균; 전문가는 자기 자신 기준의 무충돌률). ʰ는 위험 시나리오다. 2차는 새 테스트(1000000번대), 3차는 세 번째 테스트(1200000번대)이며, 두 단계의 같은 행은 학습 시드까지 같은 같은 정책이다(4.9.1절). 2차 행은 표 34와 같다. 칸별 신뢰구간은 계산하지 않은 기술 통계이며, 4차 심사에 따라 사전 등록 뒤 같은 원자료로 계산했다(`scripts/supp_round4_analysis.py`).

**(2) 확증 가설(Holm 군 A)과 K4·동등성.** 표 43은 사전 등록 가설 검정과 보조 비교다.

{{table:supp4_third_tests}}

표 43. 3단계 사전 등록 가설 검정(docs/39 3절, 세 번째 테스트; 예산 2%, 시드 10개 대응). A·B는 비교하는 두 조건의 성공률(H-L 행은 속도 오차, m/s)이고, 대괄호는 시드·에피소드 계층 부트스트랩 95% CI(10,000회)다. Holm 군 A는 H-v3·H-a3·H-a3n·H-P7·H-L이다. 'K4'는 판정 규칙(95% CI 하한 > 0)으로, 'H-K4e'는 P3·P4 제외 v4의 CARE − 트리거 혼합에 대한 90% CI와 동등 한계 ±0.03으로 판정한다. '보조' 행은 사전 등록한 보조(탐색적) 비교로 보정하지 않았다. '보조(4차 심사)' 두 행은 사전 등록 뒤 4차 심사에 따라 같은 원자료로 계산한 것이다(`scripts/supp_round4_analysis.py`). 선별 효과의 차이(H-a3 − H-a3n)는 두 비교를 같은 시드·에피소드로 대응시킨 차이의 차이이고, H-L의 계층 구간은 반사실 에피소드도 재표집한 구간이다. H-L 행의 차이는 시드 대응 차이의 시드 부트스트랩 구간이며, 단측 $p$는 P(재표집 평균 ≥ 0)이다(가설의 방향이 '< 0'이므로). 다른 행의 단측 $p$는 P(차이 ≤ 0)이며, 0.0000은 재표집 10,000회 가운데 차이 ≤ 0이 없었다는 뜻이다($p$ < 10⁻⁴, 해상도 한계). 표는 사전 등록 분석 코드의 출력(`tables/third_tests.md`)과 같은 값을 행 이름만 한국어로 바꾸어 보조 행과 함께 다시 적은 것이다.

- **H-v3(지지)**: 같은 세트에서 v4 설정은 v3 설정보다 CARE 2% 성공률이 {{s:k3:tests:H-v3:diff}} [95% CI {{s:k3:tests:H-v3:lo}}, {{s:k3:tests:H-v3:hi}}] 높았다. 2차의 H-v({{s:k4:comparisons:v4_vs_v3_care:diff}})가 재현되었다.
- **H-a3(지지)**: v4 CARE − v4 무작위는 {{s:k3:tests:H-a3:diff}} [{{s:k3:tests:H-a3:lo}}, {{s:k3:tests:H-a3:hi}}]였다(2차 H-a {{s:k4:comparisons:care_vs_random_shared:diff}}). 이 비교는 점수기 출력이 모든 방법의 학습 신호로 쓰인 v4에서의 선별 효과다. 점수기 출력을 학습에 쓰지 않은 v3 정책에서도 같은 비교(H-a)가 1단계에서 지지되었다({{s:k:comparisons:care_vs_random_shared_b0.02:diff}}, Holm 보정 $p$={{s:k:comparisons:care_vs_random_shared_b0.02:p_holm:3}}).
- **H-a3n(미지지)**: 점수기 출력을 학습에 쓰지 않은 v4(P3·P4 제외)에서 CARE − 무작위는 {{s:k3:tests:H-a3n:diff}} [{{s:k3:tests:H-a3n:lo}}, {{s:k3:tests:H-a3n:hi}}](Holm 보정 $p$={{s:k3:tests:H-a3n:p_holm:3}})였다. 사전 등록 해석 규칙에 따라, **v4 정책에서 점수기 출력을 학습에 쓰지 않는 조건에서는 CARE 선별의 고유 효과가 확인되지 않았다.** 이 정책은 2단계 탐색에서 가설을 세울 때 쓴 정책과 같은 정책이며({{s:k4:explore:k4:noaux_care_vs_random_shared:diff}}, 5.6.2절), 새로 본 것은 평가 에피소드다. 새 학습 시드의 4단계에서는 같은 가설(H-a4n)이 지지되어, 이 판정은 사전 등록 규칙에 따라 '불안정'으로 낮춘다(5.8절 (4)).
- **선별 효과의 차이(보조, 4차 심사)**: H-a3·H-a3n의 두 비교를 같은 시드·에피소드로 대응시킨 차이의 차이는 {{s:s4:ha_diff:diff}} [95% CI {{s:s4:ha_diff:lo}}, {{s:s4:ha_diff:hi}}], 단측 $p$={{s:s4:ha_diff:p_le0:3}}였다(표 43). 따라서 H-a3 지지와 H-a3n 미지지의 대비는 유의성의 차이일 뿐이며, 두 조건의 선별 효과가 다르다는 증거는 없다. 'CARE의 이득은 점수기 출력을 학습에도 쓰는 조건에서만 생긴다'는 주장은 이 자료로 할 수 없다.
- **H-P7(미지지)**: v4 CARE − 전부+P6 CARE는 {{s:k3:tests:H-P7:diff}} [{{s:k3:tests:H-P7:lo}}, {{s:k3:tests:H-P7:hi}}](Holm 보정 $p$={{s:k3:tests:H-P7:p_holm:3}})로, P7이 폐루프 성공률에 기여한다는 가설은 지지되지 않았다. H-v3({{s:k3:tests:H-v3:diff}}) 가운데 P7 몫으로 확증된 부분은 없다.
- **K4(미달)**: v4 CARE − v4 트리거 혼합은 {{s:k3:tests:K4:diff}} [{{s:k3:tests:K4:lo}}, {{s:k3:tests:K4:hi}}]로 CI 하한이 0 아래였다. 2차({{s:k4:K4:diff}})보다 점추정은 컸으나 판정은 같다.
- **H-K4e(동등 아님)**: P3·P4 제외 v4의 CARE − 트리거 혼합은 {{s:k3:tests:H-K4e:diff}}, 90% CI [{{s:k3:tests:H-K4e:lo90}}, {{s:k3:tests:H-K4e:hi90}}]로 상한이 +0.03을 넘었다. 사전 등록 해석 규칙에 따라 **차이도 동등도 확정하지 못했다.**
- **보조(판정 없음)**: v4 CARE − P3·P4 제외 v4 CARE는 {{s:k3:tests:v4_vs_noaux_care:diff}} [{{s:k3:tests:v4_vs_noaux_care:lo}}, {{s:k3:tests:v4_vs_noaux_care:hi}}]로 0을 포함하지 않았다. 점수기 출력을 보조 타깃·손실 가중으로 쓰는 것이 같은 CARE 데이터의 성공률을 올렸을 가능성과 맞는 관찰이지만, 사전 등록한 가설이 아니며 블록별로는 P3·P4 모두 검출되지 않았다((3)).

**(3) 블록별 제거(Holm 군 B).** 표 44는 v4에서 블록 하나씩을 뺀 결과다.

{{table:third_loo}}

표 44. 블록별 제거(세 번째 테스트, CARE 2%, 시드 0~4 대응). 'v4 성공률'은 같은 다섯 시드의 v4 CARE 2% 평균이다. 대괄호는 계층 부트스트랩 95% CI이고, Holm 보정은 다섯 비교를 한 군(군 B)으로 묶었다. 시드가 5개라 검정력이 낮으며, 미지지는 '효과 없음'이 아니라 '검출 못 함'이다.

- **기여 검출**: P2(시간 특징 인코더)를 빼면 {{s:k3:loo:loo_p2:diff}} [{{s:k3:loo:loo_p2:lo}}, {{s:k3:loo:loo_p2:hi}}](Holm 보정 $p$={{s:k3:loo:loo_p2:p_holm:3}}), T1(지시문 드롭아웃)을 빼면 {{s:k3:loo:loo_t1:diff}} [{{s:k3:loo:loo_t1:lo}}, {{s:k3:loo:loo_t1:hi}}]({{s:k3:loo:loo_t1:p_holm:3}}) 낮아졌다.
- **검출 못 함**: P3(보조 헤드) {{s:k3:loo:loo_p3:diff}}({{s:k3:loo:loo_p3:p_holm:3}}), P4(위험 가중 손실) {{s:k3:loo:loo_p4:diff}}({{s:k3:loo:loo_p4:p_holm:3}}), P6·P7 {{s:k3:loo:loo_p67:diff}}({{s:k3:loo:loo_p67:p_holm:3}}). P4의 점추정은 0에 가까웠다.
- 각 차이는 v4 전체 조합에서 블록 하나만 뺀 효과이므로, 블록 사이의 상호작용 때문에 더해서 H-v3와 맞추어 볼 수 없다. 블록별 몫의 합이 개선 효과를 나누어 설명한다고 해석하지 않는다.

**(4) 언어 지시 준수(K6)와 그 분해.** 표 45는 세 번째 반사실 세트의 속도 오차다.

{{table:third_k6}}

표 45. 세 번째 반사실 세트(시드 1250000번대, 63 에피소드)의 자유주행 속도 오차(CARE 2%, 시드 0~9 평균; 제어기 단독은 결정적이라 1회). 언어를 빼면 P6·P7 경로도 꺼지므로 'v4, 언어 없음'과 '전부+P6, 언어 없음'은 같은 입력이다. 'P6·P7만'은 학습 언어 임베딩·FiLM을 끄고 지시문 숫자를 결정적으로 읽는 P6·P7 경로만 남긴 정책이다.

- **K6(달성)**: 언어를 넣으면 오차가 {{s:k3:k6:K6:nolang:2}} → {{s:k3:k6:K6:lang:2}} m/s로 {{p:k3:k6:K6:speed_error_reduction}} 줄었다(시드 부트스트랩 95% 구간 [{{p:k3:k6:K6:lo}}, {{p:k3:k6:K6:hi}}], 목표 이상 비율 {{p:k3:k6:K6:p_ge_goal:2}}). 시드 {{s:k3:k6:K6:n_seeds:0}}개 모두에서 오차가 줄었다(2차는 다섯 시드 중 {{s:k4:K6:n_seeds_reduced:0}}개). 다만 시드 0({{p:k3:k6:K6:by_seed:0:1}})과 시드 5({{p:k3:k6:K6:by_seed:5:1}})의 감소율은 목표 30%에 못 미쳐, 시드 간 차이는 여전히 컸다.
- **K6 구간의 종류(4차 심사 대응).** 위 구간은 시드만 재표집한 것으로, 63개 반사실 에피소드의 변동이 빠져 있다. 시드·에피소드 계층 부트스트랩 구간은 [{{p:k3:k6:K6_hier:lo}}, {{p:k3:k6:K6_hier:hi}}](목표 이상 비율 {{p:k3:k6:K6_hier:p_ge_goal:1}}, 보조)로 더 넓지만 하한이 여전히 목표 30%를 넘었다. 같은 계산으로 2차 K6의 계층 구간은 [{{p:s4:k6_hier_stage2:lo}}, {{p:s4:k6_hier_stage2:hi}}]였다(표 23).
- **같은 정책의 시드별 값은 세트에 따라 크게 바뀌었다.** 시드 0~4의 v4 언어 있음/없음 정책은 2단계와 같은 정책인데, 시드별 감소율은 시드 0이 {{p:k4:K6:by_seed:0:1}} → {{p:k3:k6:K6:by_seed:0:1}}, 시드 2가 {{p:k4:K6:by_seed:2:1}} → {{p:k3:k6:K6:by_seed:2:1}}로 바뀌었다. 시드별 감소율의 차이는 학습 불안정보다 평가 에피소드의 변동을 주로 반영한다(5.5절 (3)).
- **H-L(미지지)**: v4(언어 있음, {{s:k3:k6:mean_speed_error:v4_lang:2}} m/s) − 'P6·P7만'({{s:k3:k6:mean_speed_error:p67_only:2}} m/s)의 시드 대응 차이는 {{s:k3:k6:H-L:diff:2}} m/s [{{s:k3:k6:H-L:lo:2}}, {{s:k3:k6:H-L:hi:2}}](Holm 보정 $p$={{s:k3:k6:H-L:p_holm:3}})로, 학습 언어 경로가 수치 목표 경로보다 오차를 더 줄인다는 근거가 없었다. 'P6·P7만'의 오차가 v4 언어 있음 이하였으므로, 사전 등록 해석 규칙에 따라 **K6 달성은 수치 목표 경로(P6·P7) 덕분이며, 학습 언어 접지는 확인되지 않았다**로 서술한다. 이는 규칙에 따른 서술일 뿐, 학습 언어 경로의 기여가 없음이 확인된 것은 아니다. H-L의 구간은 시드만 재표집해도 [{{s:k3:k6:H-L:lo:2}}, {{s:k3:k6:H-L:hi:2}}] m/s, 시드·에피소드 계층으로는 [{{s:k3:k6:H-L_hier:lo:2}}, {{s:k3:k6:H-L_hier:hi:2}}] m/s로, v4 언어 있음의 오차({{s:k3:k6:mean_speed_error:v4_lang:2}} m/s)에 견줄 만한 학습 언어 경로의 기여를 배제하지 못한다. 또 'P6·P7만'의 시드 0~4는 2단계 탐색에서 이 가설을 세울 때 쓴 정책과 같다(4.9.1절). 모든 정책을 새로 학습한 4단계에서도 H-L은 미지지였다(5.8절 (3)).
- **P7의 몫(보조)**: P7을 뺀 '전부+P6'은 언어를 넣어도 {{s:k3:k6:mean_speed_error:p6_nolang:2}} → {{s:k3:k6:mean_speed_error:p6_lang:2}} m/s로 {{p:k3:k6:p6_reduction}}만 줄어 K6 목표에 못 미쳤다. 학습 없이 제어기 $a^0$만 쓰면 {{s:k3:k6:mean_speed_error:ctrl:2}} m/s로, 언어 없는 v4 대비 {{p:s4:ctrl_reduction_stage3}} 줄었다. 즉 K6 목표(30%)에 가까운 감소가 학습 없이 손으로 쓴 규칙(지시문 숫자 읽기 + 비례 제어)만으로 얻어진다. K6의 감소 대부분은 P7(목표 속도 비례 제어 + 학습 잔차)이 있을 때 나타났다. 이 비교는 사전 등록한 보조 비교로, 판정이 없다.

**(5) AMR(K5)과 v3 대조.** 표 46은 세 번째 AMR 테스트 결과다.

{{table:third_robot}}

표 46. 세 번째 AMR 테스트 세트(시드 1300000번대, 72 에피소드)의 결과(2%는 시드 10개, 전체는 시드 3개). 선별 하이퍼파라미터는 v3 AMR 튜닝값($\lambda$=0, $\rho$=0.95)이다. v3 행은 같은 세트에서 v3 정책 설정으로 학습한 대조다.

- **K5(미달)**: 시드를 10개로 늘린 v4 CARE − 무작위는 {{s:k3:robot:K5:diff}} [95% CI {{s:k3:robot:K5:lo}}, {{s:k3:robot:K5:hi}}]로, 점추정은 0에 가까웠으나 CI 하한이 비열등 한계 −0.03보다 낮았다. 시드 10개에서도 구간 폭이 비열등을 판정하기에 넓었다(검정력 계산은 6.3절). 10개 가운데 시드 0~4는 2단계 AMR과 같은 정책이고, 새로 학습한 정책은 시드 5~9다(표 21).
- **v3 대조(A4)**: 같은 세트에서 v3 설정의 CARE − 무작위는 {{s:k3:robot:K5_v3:diff}} [{{s:k3:robot:K5_v3:lo}}, {{s:k3:robot:K5_v3:hi}}]였다. v4 CARE − v3 CARE는 {{s:k3:robot:v4_vs_v3_care:diff}} [{{s:k3:robot:v4_vs_v3_care:lo}}, {{s:k3:robot:v4_vs_v3_care:hi}}](보조, 판정 없음)로, 2단계에서 교차 세트로만 볼 수 있었던 AMR의 v3 → v4 향상이 같은 세트에서 관찰되었다.
- AMR 전체 데이터 정책과 전문가({{s:k3:robot:expert_success}})의 거리는 여전히 컸다(표 46).

**(6) 실영상 P5(탐색적, 같은 fold 재사용).** 자차 운동 이력(P5, $H_p$=8)을 넣은 v4 정책(새 정책)을 전체 데이터로 1·2단계와 같은 5-fold × 시드 2({{s:k3:comma_p5:n_runs:0}}회)로 학습·평가했다.

- 제동 시작 AUROC는 {{s:k3:comma_p5:auroc_p5}} ± {{s:k3:comma_p5:sd_p5}}로, P5가 없는 v4({{s:k3:comma_p5:auroc_v4}}, 표 38)보다 높았고 K7 목표(0.65)를 넘었다. 그러나 같은 시험 프레임의 $-a_t$ 기준선({{s:k3:comma_p5:baseline_neg_accel}})에는 못 미쳤다.
- **기반 정책이 다른 두 결과.** 폐루프 쪽 근거는 2단계 개발에서 **v3 정책에 P5만 더한** 후보의 주행 개발 세트 성공률({{s:k4:dev_choice:driving:dev_success:p5}}, 시드 3개, v3 기준 {{s:k4:dev_choice:driving:dev_success:v3}}, 5.4절 표 31)이고, 개루프 쪽 근거는 위의 **v4 정책 + P5** 실영상 AUROC다. 같은 개발 세트의 AMR에서는 v3 + P5가 v3 기준과 같았다({{s:k4:dev_choice:robot:dev_success:p5}} 대 {{s:k4:dev_choice:robot:dev_success:v3}}). 3단계 시점에는 v4 + P5를 폐루프에서 평가한 적이 없었으므로, '같은 입력이 개루프 지표는 올리고 폐루프 성공률은 낮춘다'는 상충은 서로 다른 기반 정책의 탐색적 관찰을 이은 **상충 가능성**이었다. 4단계와 함께 실행한 v4 + P5의 폐루프 개발 세트 평가(CARE 2%, 시드 0~4, 탐색)에서는 v4 + P5의 성공률({{s:p5:all_p7_p5:mean}})이 v4({{s:p5:all_p7:mean}})보다 낮아, 같은 기반 정책 설정에서도 두 방향이 관찰되었다(5.8절 (5)). 이 역시 서로 다른 도메인의 탐색적 관찰이다. 같은 fold를 재사용했고 시뮬레이터 폐루프와 실영상은 도메인이 다르므로, 이 관찰은 K7의 판정에 쓰지 않는다.

**(7) 2차 확증과 다른 점: 같은 정책, 다른 평가 세트.** 사전 등록 규칙(docs/39 4절)에 따라 두 결과를 모두 보고하고 고르지 않는다. 2·3단계의 v4·v3 정책은 학습 시드까지 같은 같은 정책이므로(4.9.1절), 아래의 차이는 모두 평가 세트(난이도와 에피소드 표집)에서 온다.

- **K2**: 2차 {{s:k4:K2:value}}(미달) → 3차 {{s:k3:kpi:K2:value}}(달성). 두 값 모두 상대 세트의 95% 구간 안에 있다(2차 [{{u:k4:K2:lo}}, {{u:k4:K2:hi}}], 3차 [{{u:k3:kpi:K2:lo}}, {{u:k3:kpi:K2:hi}}]). 이 구간은 CARE와 전체 데이터를 독립으로 재표집한 보수적 구간이며, 에피소드를 대응시키면 2차 [{{u:s4:k2_paired:stage2:lo}}, {{u:s4:k2_paired:stage2:hi}}](목표 이상 비율 {{p:s4:k2_paired:stage2:p_ge_goal:0}}), 3차 [{{u:s4:k2_paired:stage3:lo}}, {{u:s4:k2_paired:stage3:hi}}]({{p:s4:k2_paired:stage3:p_ge_goal:0}})로 좁아지지만 두 구간 모두 여전히 목표 0.90을 포함한다(보조, 4.2절).
- **K3·K6의 여유**: K3 {{s:k4:K3:value}} → {{s:k3:kpi:K3:value}}, K6 {{p:k4:K6:speed_error_reduction}}(감소 시드 {{s:k4:K6:n_seeds_reduced:0}}/5) → {{p:k3:k6:K6:speed_error_reduction}}(감소 시드 {{s:k3:k6:K6:n_seeds_reduced:0}}/{{s:k3:k6:K6:n_seeds:0}}).
- **세트 난이도(확정된 설명).** 같은 정책 넷의 성공률이 세 번째 세트에서 함께 높았다. v4 CARE {{s:k4:driving:v4:care:success}} → {{s:k3:tests:H-a3:a}}, 트리거 혼합 {{s:k4:driving:v4:mix_trigger:success}} → {{s:k3:tests:K4:b}}, 무작위 {{s:k4:driving:v4:random_shared:success}} → {{s:k3:tests:H-a3:b}}, v3 CARE {{s:k4:driving:v3:care:success}} → {{s:k3:tests:H-v3:b}}. 전문가 자신의 성공률도 주행 {{s:s4:expert:stage2:driving}} → {{s:s4:expert:stage3:driving}}, 위험 시나리오 {{s:s4:expert:stage2:hazard}} → {{s:s4:expert:stage3:hazard}}, 보행자 횡단 {{s:s4:expert:stage2:vru_crossing}} → {{s:s4:expert:stage3:vru_crossing}}으로 높았다(표 42). 정책이 같으므로 세 번째 세트가 두 번째보다 쉬웠던 것이며, 2차·3차의 KPI 차이를 v4 정책의 향상으로 읽지 않는다. 세트 난이도는 같은 세트 안의 대응 비교(H-v3, H-a3)에서는 두 조건에 함께 작용하지만, 점추정 기준 KPI(K1~K3, K6)의 달성 여부는 직접 바꿀 수 있다. 다만 세 세트 가운데 세 번째 세트가 가장 쉬운 것은 아니다. 1단계 세트의 v3 CARE(1단계 스위트로 학습해 특징 프리롤 처리만 다른 정책)는 {{s:k:K1:value}}로 세 번째 세트의 v3 CARE({{s:k3:tests:H-v3:b}})보다 높았다.
- **2·3단계 종합(기술 통계, 4차 심사 대응).** 같은 정책을 두 세트에서 평가했으므로, 두 세트의 에피소드(합계 {{s:s4:pooled_23:n_episodes:0}}개)를 합쳐 시드는 공유하고 에피소드는 세트 안에서 재표집한 종합 값을 함께 적는다. K1 {{s:s4:pooled_23:K1:value}} [{{u:s4:pooled_23:K1:lo}}, {{u:s4:pooled_23:K1:hi}}](목표 이상 비율 {{p:s4:pooled_23:K1:p_ge_goal:0}}), K2 {{s:s4:pooled_23:K2:value}} [{{u:s4:pooled_23:K2:lo}}, {{u:s4:pooled_23:K2:hi}}]({{p:s4:pooled_23:K2:p_ge_goal:1}}), K3 {{s:s4:pooled_23:K3:value}} [{{u:s4:pooled_23:K3:lo}}, {{u:s4:pooled_23:K3:hi}}]({{p:s4:pooled_23:K3:p_ge_goal:1}}). K1·K3은 두 세트를 합쳐도 목표를 넘지만, K2의 종합 점추정은 목표와 거의 같아 **판정이 불확정**이다. 이 종합 값은 사전 등록 판정이 아니며, 판정은 단계별로 그대로 둔다. 두 세트의 정책이 같으므로 종합 값은 '더 넓은 평가 에피소드 위의 같은 정책'의 성능을 요약할 뿐, 학습 시드 표집을 늘린 것이 아니다.
- **같은 판정**: K4 미달, K5 미달, H-v·H-a 지지, 'K6은 P6·P7 경로의 결과'라는 서술은 같은 학습 정책이 서로 다른 두 평가 세트에서 같은 판정을 받은 것이다. 이를 독립된 증거 두 개로 세지 않는다. 학습 시드 표집에 대한 독립 반복은 4단계(4.10절, 5.8절)가 맡았고, 그 결과 이 판정들은 새 학습 시드에서도 같았다(5.8절 (4), 표 50).

### 5.8 4단계(새 학습 시드·네 번째 평가 세트): 독립 반복

4단계는 사전 등록(docs/40, 커밋 `6eff953`, 2026-10-07 00:41:23 UTC)을 커밋한 뒤 00:41:25 UTC에 시작했다. 확증 실행(`fourth`)은 04:01:33 UTC, 분석(`fourthreport`)은 04:01:37 UTC, 탐색(`p5dev`)은 04:14:59 UTC에 끝났다. 중단 기록은 없고, 미측정으로 남은 조건도 없다. 설계와 해석 규칙은 4.10절에 있다. 확증 조건의 정책 115개는 모두 학습 시드 10~19로 새로 학습했으므로 2·3단계 정책과 겹치지 않는다. 평가 세트(주행 test4 1400000번대 147 에피소드, 반사실 cf4 1450000번대 63 에피소드)도 처음 보는 세트다. 따라서 4단계는 학습 시드 표집(공유 저장소·초기화·미니배치)과 평가 에피소드를 모두 새로 뽑은 독립 반복이다. 새 세트의 전문가 주행 성공률은 {{s:k5:expert_success}}이다. AMR과 실영상은 사전 등록대로 넣지 않았으므로, 4단계는 K5·K7을 판정하지 않는다. 이 절의 결과는 (5)의 탐색을 빼면 모두 사전 등록 확증이다.

**(1) KPI(주행 K1~K3).** 표 47은 네 번째 평가 세트의 주행 결과다.

{{table:fourth_driving}}

표 47. 4단계 네 번째 평가 세트(test4, 시드 1400000번대, 147 에피소드)의 주행 결과(평균 ± 시드 표준편차; 학습 시드 10~19, 전체 100%는 10~14). 'P3·P4 제외'는 점수기 출력을 학습에 쓰지 않는 v4, '전부+P6'은 P7을 뺀 조합이다. 위험 시나리오 성공률은 시드 평균이다. 선별·풀·점수기는 모든 행이 같고, 모든 행이 2·3단계에 없던 새 정책이다.

- **K1(달성)**: v4 CARE 2%의 성공률은 {{s:k5:kpi:K1:value}}(시드 {{s:k5:kpi:K1:n_seeds:0}}개), 시드·에피소드 부트스트랩 95% 구간 [{{u:k5:kpi:K1:lo}}, {{u:k5:kpi:K1:hi}}], 재표집에서 목표 이상 비율 {{p:k5:kpi:K1:p_ge_goal:2}}였다.
- **K2(달성)**: 전체 데이터({{s:k5:kpi:K2:full}}, 시드 10~14)에 대한 비율은 {{s:k5:kpi:K2:value}} [{{u:k5:kpi:K2:lo}}, {{u:k5:kpi:K2:hi}}]로 점추정이 목표 0.90을 넘었다. 목표 이상 비율은 {{p:k5:kpi:K2:p_ge_goal:0}}로 구간이 목표를 넓게 포함하므로 경계 달성이다(구간은 CARE와 전체를 독립으로 재표집한 보수적 구간, 4.2절).
- **K3(달성)**: 위험 시나리오 성공률은 {{s:k5:kpi:K3:value}} ± {{s:k5:kpi:K3:sd}}(시드 표준편차), 95% 구간 [{{u:k5:kpi:K3:lo}}, {{u:k5:kpi:K3:hi}}], 목표 이상 비율 {{p:k5:kpi:K3:p_ge_goal}}였다.
- **K1·K3는 여전히 선별법을 판별하지 않는다.** 같은 v4 정책에서 무작위 2%({{s:k5:tests:H-a3:b}})와 트리거 혼합({{s:k5:tests:K4:b}})도 K1 목표를 넘었고, 두 방법의 위험 시나리오 성공률도 K3 목표를 넘었다(표 47).
- 2·3단계와 4단계의 값 차이는 평가 세트(난이도·에피소드)와 학습 시드 표집이 함께 바뀐 결과이므로, 개선이나 악화로 읽지 않는다. 같은 세트 안의 대응 비교만 아래 (2)~(4)에서 판정한다.

**(2) 확증 가설(Holm 군 A)과 K4·동등성.** 표 48은 사전 등록 가설 검정과 보조 비교다.

{{table:fourth_tests}}

표 48. 4단계 사전 등록 가설 검정(docs/40 3절, 네 번째 평가 세트; 예산 2%, 학습 시드 10~19 대응). 표는 사전 등록 분석 코드(3단계와 같은 함수)의 출력이며, 행 이름은 3단계 코드의 식별자를 그대로 쓴다. 4단계에서 각 행의 뜻은 다음과 같다. 'H-v3'은 H-v4(v4 CARE − v3 CARE), 'H-a3'은 H-a4(v4 CARE − v4 무작위), 'H-a3n'은 H-a4n(P3·P4 제외 v4의 CARE − 무작위), 'H-P7'은 H-P7b(v4 CARE − 전부+P6 CARE), 'H-L'은 H-L4(cf4에서 v4 언어 있음 − 'P6·P7만'의 속도 오차, m/s)다. 'K4'는 v4 CARE − v4 트리거 혼합, 'mix_vs_random_v4'와 'mix_vs_random_noaux'는 트리거 혼합 − 무작위(v4, P3·P4 제외 v4), 'v4_vs_noaux_care'는 v4 CARE − P3·P4 제외 v4 CARE(이 세 행은 보조, 보정 없음), 'H-K4e'는 P3·P4 제외 v4의 CARE − 트리거 혼합의 90% CI와 동등 한계 ±0.03이다. Holm 군 A는 H-v4·H-a4·H-a4n·H-P7b·H-L4다. 대괄호는 시드·에피소드 계층 부트스트랩 95% CI(10,000회; H-L 행은 시드 대응 차이의 시드 부트스트랩)다. H-L 행의 단측 $p$는 P(재표집 평균 ≥ 0)이고(가설 방향이 '< 0'), 다른 행은 P(차이 ≤ 0)이며, 0.0000은 재표집 10,000회 가운데 차이 ≤ 0이 없었다는 뜻이다($p$ < 10⁻⁴).

- **H-v4(지지)**: 같은 세트에서 v4 설정은 v3 설정보다 CARE 2% 성공률이 {{s:k5:tests:H-v3:diff}} [95% CI {{s:k5:tests:H-v3:lo}}, {{s:k5:tests:H-v3:hi}}] 높았다(Holm 보정 $p$ < 10⁻⁴). 새로 학습한 정책에서도 정책 설정의 개선이 확인되었다.
- **H-a4(지지)**: v4 CARE − v4 무작위는 {{s:k5:tests:H-a3:diff}} [{{s:k5:tests:H-a3:lo}}, {{s:k5:tests:H-a3:hi}}](Holm 보정 $p$={{s:k5:tests:H-a3:p_holm:3}})였다.
- **H-a4n(지지)**: 점수기 출력을 학습에 쓰지 않은 v4(P3·P4 제외)에서 CARE − 무작위는 {{s:k5:tests:H-a3n:diff}} [{{s:k5:tests:H-a3n:lo}}, {{s:k5:tests:H-a3n:hi}}](Holm 보정 $p$={{s:k5:tests:H-a3n:p_holm:3}})였다. 3단계의 H-a3n(미지지, {{s:k3:tests:H-a3n:diff}}, Holm 보정 $p$={{s:k3:tests:H-a3n:p_holm:3}})과 판정이 다르다((4)).
- **H-P7b(미지지)**: v4 CARE − 전부+P6 CARE는 {{s:k5:tests:H-P7:diff}} [{{s:k5:tests:H-P7:lo}}, {{s:k5:tests:H-P7:hi}}](Holm 보정 $p$={{s:k5:tests:H-P7:p_holm:3}})로, P7이 폐루프 성공률에 기여한다는 가설은 다시 지지되지 않았다.
- **K4(미달)**: v4 CARE − v4 트리거 혼합은 {{s:k5:tests:K4:diff}} [{{s:k5:tests:K4:lo}}, {{s:k5:tests:K4:hi}}]로 CI 하한이 0 아래였다(단측 $p$={{s:k5:tests:K4:p_le0:3}}).
- **H-K4e(동등 아님)**: P3·P4 제외 v4의 CARE − 트리거 혼합은 {{s:k5:tests:H-K4e:diff}}, 90% CI [{{s:k5:tests:H-K4e:lo90}}, {{s:k5:tests:H-K4e:hi90}}]로 상한이 +0.03을 넘었다. 사전 등록 해석 규칙에 따라 차이도 동등도 확정하지 못했다.
- **보조(판정 없음)**: 트리거 혼합 − 무작위는 v4에서 {{s:k5:tests:mix_vs_random_v4:diff}} [{{s:k5:tests:mix_vs_random_v4:lo}}, {{s:k5:tests:mix_vs_random_v4:hi}}]로 0을 포함하지 않았고(3단계 {{s:k3:tests:mix_vs_random_v4:diff}} [{{s:k3:tests:mix_vs_random_v4:lo}}, {{s:k3:tests:mix_vs_random_v4:hi}}]), P3·P4 제외 v4에서는 {{s:k5:tests:mix_vs_random_noaux:diff}} [{{s:k5:tests:mix_vs_random_noaux:lo}}, {{s:k5:tests:mix_vs_random_noaux:hi}}]였다. 감속 트리거로 채운 위험 지향 몫도 v4에서 무작위 몫보다 나았을 가능성과 맞는 관찰이며, K4 미달과 함께 읽으면 H-a의 이득을 맥락 점수기 고유의 것으로 볼 근거는 4단계에서도 없다. v4 CARE − P3·P4 제외 v4 CARE는 {{s:k5:tests:v4_vs_noaux_care:diff}} [{{s:k5:tests:v4_vs_noaux_care:lo}}, {{s:k5:tests:v4_vs_noaux_care:hi}}]로, 3단계({{s:k3:tests:v4_vs_noaux_care:diff}} [{{s:k3:tests:v4_vs_noaux_care:lo}}, {{s:k3:tests:v4_vs_noaux_care:hi}}])와 달리 0을 포함했다.

**(3) 언어 지시 준수(K6)와 H-L4.** 표 49는 네 번째 반사실 세트의 속도 오차다.

{{table:fourth_k6}}

표 49. 네 번째 반사실 세트(cf4, 시드 1450000번대, 63 에피소드)의 자유주행 속도 오차(CARE 2%, 학습 시드 10~19 평균). 언어를 빼면 P6·P7 경로도 꺼진다. 'P6·P7만'은 학습 언어 임베딩·FiLM을 끄고 지시문 숫자를 결정적으로 읽는 P6·P7 경로만 남긴 정책이다. 3단계(표 45)와 달리 전부+P6과 제어기 단독은 4단계 실행 조건에 넣지 않았다(docs/40 2절).

- **K6(달성)**: 언어를 넣으면 오차가 {{s:k5:k6:K6:nolang:2}} → {{s:k5:k6:K6:lang:2}} m/s로 {{p:k5:k6:K6:speed_error_reduction}} 줄었다(시드 부트스트랩 95% 구간 [{{p:k5:k6:K6:lo}}, {{p:k5:k6:K6:hi}}], 목표 이상 비율 {{p:k5:k6:K6:p_ge_goal:2}}). 시드 {{s:k5:k6:K6:n_seeds:0}}개 모두에서 오차가 줄었으나, 시드 10({{p:k5:k6:K6:by_seed:10:1}})과 시드 12({{p:k5:k6:K6:by_seed:12:1}})의 감소율은 목표 30%에 못 미쳤다.
- **K6 계층 구간(보조).** 시드·에피소드 계층 부트스트랩 구간은 [{{p:k5:k6:K6_hier:lo}}, {{p:k5:k6:K6_hier:hi}}](목표 이상 비율 {{p:k5:k6:K6_hier:p_ge_goal:1}})로, 3단계([{{p:k3:k6:K6_hier:lo}}, {{p:k3:k6:K6_hier:hi}}])와 달리 하한이 목표 30% 아래였다. 판정(점추정 기준 달성)은 사전 등록대로 두지만, 4단계의 K6 달성은 반사실 에피소드의 변동을 감안하면 목표를 확실히 넘었다고 할 수 없다.
- **H-L4(미지지, 반대 방향).** v4(언어 있음, {{s:k5:k6:mean_speed_error:v4_lang:2}} m/s) − 'P6·P7만'({{s:k5:k6:mean_speed_error:p67_only:2}} m/s)의 시드 대응 차이는 {{s:k5:k6:H-L:diff:2}} m/s [{{s:k5:k6:H-L:lo:2}}, {{s:k5:k6:H-L:hi:2}}](Holm 보정 $p$={{s:k5:k6:H-L:p_holm:3}})였다. 학습 언어 경로를 더한 v4의 오차가 오히려 컸고, 시드 부트스트랩 구간은 0을 포함하지 않았다. 시드·에피소드 계층 구간은 [{{s:k5:k6:H-L_hier:lo:2}}, {{s:k5:k6:H-L_hier:hi:2}}] m/s로 0을 조금 포함한다(보조). 반대 방향(학습 언어 경로가 오차를 키움)은 사전 등록한 가설이 아니므로 판정하지 않고 방향만 보고한다.
- **해석.** 'P6·P7만'의 오차가 v4 언어 있음 이하였으므로, 3단계와 같은 해석 규칙(docs/39 4절)이 다시 적용되어 **K6 달성은 수치 목표 경로(P6·P7) 덕분이며 학습 언어 접지는 확인되지 않았다**로 서술한다. 3단계의 H-L은 'P6·P7만'의 시드 0~4가 가설을 세운 정책과 같았으나(4.9.1절), 4단계의 정책은 모두 새로 학습한 것이다. 학습 언어 경로가 수치 목표 경로보다 오차를 더 줄인다는 근거는 서로 다른 학습 시드의 두 확증에서 모두 없었고, 4단계 점추정은 그 반대 방향이었다.

**(4) 3·4단계 대조와 해석 규칙 적용.** 표 50은 3단계와 4단계의 같은 지표·가설을 나란히 놓고, docs/40 4절의 해석 규칙을 적용한 결과다.

| 지표·가설 | 3단계(세 번째 테스트, 학습 시드 0~9) | 4단계(네 번째 평가, 학습 시드 10~19) | 판정(3단계 / 4단계) | 해석 규칙 적용(docs/40 4절) |
|---|---|---|---|---|
| K1 CARE 2% 성공률(≥ 0.80) | {{s:k3:kpi:K1:value}} [{{u:k3:kpi:K1:lo}}, {{u:k3:kpi:K1:hi}}] | {{s:k5:kpi:K1:value}} [{{u:k5:kpi:K1:lo}}, {{u:k5:kpi:K1:hi}}] | 달성 / 달성 | 독립 학습 시드에서 재현 |
| K2 데이터 효율(≥ 0.90) | {{s:k3:kpi:K2:value}}(목표 이상 {{p:k3:kpi:K2:p_ge_goal:0}}) | {{s:k5:kpi:K2:value}}(목표 이상 {{p:k5:kpi:K2:p_ge_goal:0}}) | 달성 / 달성 | 재현. 단 두 번 모두 경계 달성이고 2단계는 미달 |
| K3 위험 시나리오(≥ 0.75) | {{s:k3:kpi:K3:value}} [{{u:k3:kpi:K3:lo}}, {{u:k3:kpi:K3:hi}}] | {{s:k5:kpi:K3:value}} [{{u:k5:kpi:K3:lo}}, {{u:k5:kpi:K3:hi}}] | 달성 / 달성 | 재현 |
| K4 CARE − 트리거 혼합(CI 하한 > 0) | {{s:k3:tests:K4:diff}} [{{s:k3:tests:K4:lo}}, {{s:k3:tests:K4:hi}}] | {{s:k5:tests:K4:diff}} [{{s:k5:tests:K4:lo}}, {{s:k5:tests:K4:hi}}] | 미달 / 미달 | 재현(미달) |
| K6 속도 오차 감소(≥ 30%) | {{p:k3:k6:K6:speed_error_reduction}}; 계층 [{{p:k3:k6:K6_hier:lo}}, {{p:k3:k6:K6_hier:hi}}] | {{p:k5:k6:K6:speed_error_reduction}}; 계층 [{{p:k5:k6:K6_hier:lo}}, {{p:k5:k6:K6_hier:hi}}] | 달성 / 달성 | 재현. 4단계 계층 하한은 목표 아래(보조) |
| H-v: v4 CARE − v3 CARE | {{s:k3:tests:H-v3:diff}} [{{s:k3:tests:H-v3:lo}}, {{s:k3:tests:H-v3:hi}}] | {{s:k5:tests:H-v3:diff}} [{{s:k5:tests:H-v3:lo}}, {{s:k5:tests:H-v3:hi}}] | 지지 / 지지 | 재현 |
| H-a: v4 CARE − v4 무작위 | {{s:k3:tests:H-a3:diff}} [{{s:k3:tests:H-a3:lo}}, {{s:k3:tests:H-a3:hi}}] | {{s:k5:tests:H-a3:diff}} [{{s:k5:tests:H-a3:lo}}, {{s:k5:tests:H-a3:hi}}] | 지지 / 지지 | 재현 |
| H-a(noaux): P3·P4 제외 v4의 CARE − 무작위 | {{s:k3:tests:H-a3n:diff}} [{{s:k3:tests:H-a3n:lo}}, {{s:k3:tests:H-a3n:hi}}], Holm $p$ {{s:k3:tests:H-a3n:p_holm:3}} | {{s:k5:tests:H-a3n:diff}} [{{s:k5:tests:H-a3n:lo}}, {{s:k5:tests:H-a3n:hi}}], Holm $p$ {{s:k5:tests:H-a3n:p_holm:3}} | 미지지 / 지지 | **불일치 → 불안정** |
| H-P7: v4 CARE − 전부+P6 CARE | {{s:k3:tests:H-P7:diff}} [{{s:k3:tests:H-P7:lo}}, {{s:k3:tests:H-P7:hi}}] | {{s:k5:tests:H-P7:diff}} [{{s:k5:tests:H-P7:lo}}, {{s:k5:tests:H-P7:hi}}] | 미지지 / 미지지 | 재현(미지지) |
| H-L: v4 언어 − P6·P7만(m/s, < 0 가설) | {{s:k3:k6:H-L:diff:2}} [{{s:k3:k6:H-L:lo:2}}, {{s:k3:k6:H-L:hi:2}}] | {{s:k5:k6:H-L:diff:2}} [{{s:k5:k6:H-L:lo:2}}, {{s:k5:k6:H-L:hi:2}}] | 미지지 / 미지지 | 재현(미지지). 4단계는 반대 방향 |
| H-K4e: P3·P4 제외 v4의 CARE − 트리거 혼합(±0.03, 90% CI) | {{s:k3:tests:H-K4e:diff}} [{{s:k3:tests:H-K4e:lo90}}, {{s:k3:tests:H-K4e:hi90}}] | {{s:k5:tests:H-K4e:diff}} [{{s:k5:tests:H-K4e:lo90}}, {{s:k5:tests:H-K4e:hi90}}] | 동등 아님 / 동등 아님 | 재현(차이도 동등도 미확정) |

표 50. 3단계와 4단계의 판정 대조. 대괄호는 K1~K3이 시드·에피소드 부트스트랩 95% 구간, K4·H-v·H-a·H-a(noaux)·H-P7이 계층 부트스트랩 95% CI, K6 '계층'이 시드·에피소드 계층 95% 구간(보조), H-L이 시드 부트스트랩 95% 구간, H-K4e가 계층 부트스트랩 90% CI다. 두 단계는 평가 세트와 학습 시드가 모두 다르므로 값의 차이를 비교하지 않고 판정의 일치만 본다. K5·K7은 4단계에 넣지 않았다(4.10절).

- **(규칙 1, 일치) 독립 학습 시드에서 재현된 것.** K1·K2·K3·K6 달성, K4 미달, H-v·H-a 지지, H-P7·H-L 미지지, H-K4e 동등 아님. 3단계의 이 판정들은 2단계와 같은 학습 정책에서 나온 것이었으나(4.9.1절), 4단계는 학습 시드까지 새로 뽑은 정책에서 같은 판정을 얻었다. 다만 K2는 두 번 모두 목표 이상 비율이 70% 안팎인 경계 달성이고(2단계는 미달), K6의 4단계 계층 구간 하한은 목표 아래다. 이 둘은 판정이 재현되었을 뿐, 목표를 넘는 여유가 확인된 것은 아니다.
- **(규칙 1, 불일치) H-a(noaux)는 불안정.** 점수기 출력을 학습에 쓰지 않은 v4의 CARE − 무작위는 3단계에서 미지지, 4단계에서 지지였다. 사전 등록 규칙에 따라 두 결과를 모두 보고하고, "점수기 출력 없이 학습할 때의 CARE 선별 효과"는 **불안정**으로 쓴다. 즉 이 조건에서 효과가 있다고도, 없다고도 확정하지 않는다. 두 단계의 점추정은 모두 양수였다(3단계 {{s:k3:tests:H-a3n:diff}}, 4단계 {{s:k5:tests:H-a3n:diff}}).
- **(규칙 2) 적용되지 않음.** 규칙 2는 H-a4 지지·H-a4n 미지지일 때의 서술 제한이다. 4단계에서는 둘 다 지지되었으므로 이 규칙의 전제가 성립하지 않는다. 따라서 "CARE의 이득은 점수기 출력을 학습에도 쓰는 조건에서만 생긴다"는 서술은 3단계의 차이의 차이(5.7절 (2), 보조)에 이어 4단계에서도 근거가 없다. 4단계에서는 두 선별 효과의 차이를 검정하지 않았다.
- **K6 메커니즘.** H-L이 두 확증에서 모두 미지지였고 4단계는 반대 방향이었으므로, "K6 달성은 P6·P7 경로의 결과이며 학습 언어 접지는 확인되지 않았다"는 서술은 독립 학습 시드에서 재현되었다. 학습 언어 경로가 오차를 키운다는 주장은 사전 등록 가설이 아니고 계층 구간이 0을 포함하므로 하지 않는다.

**(5) v4 + P5의 폐루프 성공률(탐색, 4차 심사 N3).** 주행 개발 세트(160000번대, 2단계 개발에 이미 쓴 세트)에서 CARE 2%, 시드 0~4로 v4 + P5(자차 운동 이력 $H_p$=8)와 v4를 평가했다. 성공률은 v4 + P5 {{s:p5:all_p7_p5:mean}} ± {{s:p5:all_p7_p5:sd}}, v4 {{s:p5:all_p7:mean}} ± {{s:p5:all_p7:sd}}(시드 표준편차, 각 {{s:p5:all_p7:n:0}}개)로, 다섯 시드 모두에서 v4 + P5가 v4보다 낮았다.

- 같은 기반 정책 설정(v4)에서 P5는 시뮬레이터 폐루프 성공률을 낮췄고, 실영상 개루프 AUROC는 높였다({{s:k3:comma_p5:auroc_v4}} → {{s:k3:comma_p5:auroc_p5}}, 5.7절 (6)). 3단계까지는 기반 정책이 다른 두 결과(v3 + P5의 폐루프, v4 + P5의 개루프)를 이은 '상충 가능성'이었으나(4차 심사 N3), 이제 같은 기반 정책 설정에서 두 방향이 관찰되었다.
- 그러나 두 관찰은 모두 탐색이고 서로 다른 도메인이다. 폐루프 쪽은 이미 개발에 쓴 세트이며, v4는 이 세트에서 성공률 최댓값으로 고른 조합이라 같은 세트의 비교는 v4에 유리할 수 있다. 개루프 쪽은 같은 fold를 재사용했고, 학습 데이터(시뮬레이터 대 실주행)가 달라 두 정책은 설정만 같고 가중치는 다르다. 신뢰구간이나 검정은 계산하지 않았다. 따라서 이 관찰은 '같은 입력이 개루프 지표는 올리고 폐루프 성공률은 낮출 수 있다'는 상충을 보강할 뿐 확증하지 않으며, K7 판정에 쓰지 않는다(6.4절).

### 5.9 계산 비용

표 51은 큐레이션과 하위 정책의 계산 비용이다. 점수기는 예비 연구·v3·v4에서 같으므로 예비 연구의 측정값을 그대로 쓴다.

| 항목 | 값 | 비고 |
|---|---|---|
| 맥락 점수기 파라미터 | {{s:cost_scorer_params:0}} | 멀티채널 CNN-GRU, 창 16. v3·v4에서 재학습하지 않음 |
| 점수기 1회 추론(창 1개, 배치 1) | {{s:cost_scorer_onnx_ms:3}} ms(ONNX Runtime) / {{s:cost_scorer_torch_ms:2}} ms(PyTorch) | x86 CPU 1스레드 |
| 검출기 YOLOv8n(참고) | {{s:cost_yolo640_ms:1}} ms(640) / {{s:cost_yolo320_ms:1}} ms(320) | x86 CPU 4스레드 |
| 하위 정책 파라미터 | v3 {{s:k4:cost:v3:n_params:0}} / v4 {{s:k4:cost:v4:n_params:0}} | v4 증가분은 GRU 특징 인코더, 보조 헤드, P6 MLP와 헤드 입력 확장이다. P7은 학습 파라미터가 없고, 보조 헤드는 추론에 쓰지 않는다 |
| 정책 학습 처리량(통제 측정) | v3 1,657 / 1,579, v4 1,466 / 1,433 samples/s | 1스레드, 배치 128, 같은 기계에서 교대 2회 측정. v4는 P6·P7을 넣기 전 조합(564,526 파라미터). 출처: `experiments/exp_130_vla_v4/summary/a14_policy_blocks/bench_1thread.json` |
| 정책 학습 처리량(본 실험 기록) | v3 {{s:k4:cost:v3:samples_per_s:0}}, v4 {{s:k4:cost:v4:samples_per_s:0}} samples/s(중앙값) | 새 테스트 2% 학습의 기록. 작업자 4개가 CPU를 공유한 상태라 참고치다 |

표 51. 계산 비용(x86 클라우드 CPU). 배포 대상 임베디드 장치(Jetson Orin Nano)에서의 지연·전력은 측정하지 않았다.

- 표 51은 통제 측정에서 v4 정책 학습이 v3보다 약 10% 느렸음을 보여 준다. 특징 이력 조회, GRU의 8단계 순차 계산, 보조 헤드 역전파가 원인이다.
- 수집 시점 큐레이션의 비용(점수기)은 v3·v4에서 바뀌지 않았다. v4의 추가 비용은 모두 학습·추론하는 정책 쪽에 있다.

### 5.10 예비 연구(v2)의 설계와 결과 요약

예비 연구는 v3 이전의 벤치마크와 정책(특징 토큰 없음, 519,336 파라미터)으로 수행했다. 원자료에서 찾은 결함과 수정은 3.6.3절(표 5)에 있으며, 여기서는 본 연구의 판단에 쓰인 설계와 결과만 요약한다.

**설계.** 주행 큐레이션 풀은 {{s:pool_episodes:0}} 에피소드({{s:pool_frames:0}} 프레임)였고, GT 위험 프레임은 전체의 {{p:pool_hazard_ratio}}였다. 주 예산 2%에서는 표 7의 10개 선별법 전부를, 예산 1·5·10%에서는 핵심 6개 방법(무작위, 감속 트리거, 불확실성, 맥락 이벤트, 오라클, CARE)을 비교했다. 시드는 3개, 테스트는 147 에피소드(시드 200000번대)였다. 설계의 일부는 파일럿 결과를 본 뒤 정해졌다. 모두 검증 세트로 결정했으므로 테스트 누출은 아니지만, 이후의 1~2% 예산 비교를 확증적 결과로 볼 수 없게 만든 요인이다.

- **학습 단계 수**: 전체 데이터의 검증 성공률이 $T$=3000에서 {{s:pilot:full:1.00:3000:success}}, 6000에서 {{s:pilot:full:1.00:6000:success}}여서, 데이터가 많은 조건만 불리해지는 교란을 피하려고 $T$=6000으로 정했다.
- **성공 정의**: 멈춰 서는 정책이 충돌률 0을 얻는 것을 보고, 성공을 '무충돌'에서 '무충돌 ∧ 진행 ≥ 0.8×전문가'로 바꾸었다.
- **예산 범위**: 무작위 선택의 검증 성공률이 10%에서 이미 {{s:pilot:random:0.10:6000:success}}에 이르러(천장 효과 우려), 예산을 1·2%까지 낮추었다.
- **CARE 하이퍼파라미터**: 처음 설정한 $\rho$=0.3은 정상 주행을 무너뜨렸다(2% 검증 성공률 {{s:pilot:ours:0.02:6000:success}}). $\lambda\in\{0,0.5\}$, $\rho\in\{0.25,0.5,0.75,0.9\}$ 격자에서 $\lambda$=0.5, $\rho$=0.9를 골랐는데, 최적점이 격자 경계에 있었다.
- **시드 수**: 파일럿의 시드 표준편차로 시드 3개를 정했으나, 주 실험(예산 2%)에서 관찰한 무작위 선택의 시드 표준편차({{s:main_sd_random_b0.02}})는 파일럿보다 3배 이상 컸다. 시드 3개의 최소 검출 효과는 약 {{p:main_mde_3seeds}}p로, 관찰된 2% 차이({{s:hb:ours_vs_random_b0.02:diff}})보다 컸다.
- **AMR**: 주행 설정을 그대로 옮긴 CARE가 AMR 테스트에서 무작위보다 낮은 것을 본 뒤 AMR 검증 세트로 재탐색했다(같은 설정이 다시 선택됨). v3에서는 AMR 튜닝을 테스트 전에 하도록 사전 등록했다(4.5절).
- **확증 실험**: `docs/32_확증실험_사전등록.md`(커밋 `fcd7f3f`)로 새 학습 시드 7개(3~9)와 새 평가 세트(시드 700000번대, 147 에피소드)에서 예산 2%의 CARE − 무작위(H1, 1차), 예산 1%의 같은 비교(H2), 같은 비율의 저장소에 다른 점수를 섞은 혼합 통제(H3a 감속 트리거, H3b 오라클, H3c 불확실성 단독)를 검정했다. 혼합 통제는 저장소 클립을 공유하지 않는 비중첩 설계였다.

**결과.**

- **위험 편향 붕괴**: 위험 표본을 최대화하는 선별(감속 트리거, 맥락 이벤트, 오라클, 정책 손실 기반 오프라인 선별)은 주행 예산 2%에서 성공률 {{s:risk_methods_max_success_b0.02:2}} 이하로 무너졌다. 이 결과는 1~10% 예산과 AMR에서도 재현되었고, v3의 감속 트리거 단독(5.2절)에서도 다시 나타났다.
- **CARE − 무작위**: 시드 3개 탐색 실험에서는 신뢰구간이 0을 포함했으나({{s:hb:ours_vs_random_b0.02:diff}} [95% CI {{s:hb:ours_vs_random_b0.02:lo}}, {{s:hb:ours_vs_random_b0.02:hi}}]), 사전 등록 확증 실험에서는 예산 2%의 우위가 지지되었다({{s:ct:H1_ours_vs_random_b0.02:diff}} [95% CI {{s:ct:H1_ours_vs_random_b0.02:lo}}, {{s:ct:H1_ours_vs_random_b0.02:hi}}]). 예산 1%의 우위와 혼합 통제군 대비 우위는 유의하지 않았다(Holm 보정 $p$ ≥ {{s:ct:H3a_ours_vs_mix_trigger:p_holm:2}}).
- **개루프와 폐루프**: 전체 구간 개루프 MAE는 폐루프 성공률과 강하게 상관했지만($r_s$={{s:ol_cl_spearman_mae:2}}), 예산 2% 안에서 위험 구간 MAE는 성공률과 오히려 양(+)의 상관을 보였다($r_s$={{s:ol_cl_within2_mae_hazard:2}}, $p$={{s:ol_cl_within2_mae_hazard_p:3}}).
- **판단하지 못한 것**: AMR에서 CARE({{s:robot:ours:0.02:success}})와 무작위({{s:robot:random:0.02:success}})의 차이는 유의하지 않았다. 실영상 제동 시작 AUROC는 전체 데이터에서도 {{s:comma:full:1.00:brake_onset_auroc:2}}에 그쳤다. 언어 효과는 지시 정보가 초기 속도로 누출되어 측정하지 못했다.

이 결과가 K1~K7의 목표값과 v3의 벤치마크 수정(B1~B3), 공유 저장소 선별(M1)의 근거가 되었다(3.6.3절, 3.7절).

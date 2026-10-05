## 5. 실험 결과

결과는 연구 단계 순서로 보고한다. 5.1절은 핵심 지표(KPI) K1~K7의 판정을 두 확증 단계에 걸쳐 요약한다. 5.2절은 1차 확증(v3, 사전 등록 docs/34, 테스트 시드 800000번대)의 결과다. 5.3절은 v3 원자료로 미달 원인을 진단한 사후 분석이고, 5.4절은 그 진단에 따라 정책·학습 블록을 고른 v4 개발 세트 결과다. 5.5절은 처음 보는 새 테스트 세트(시드 1000000번대)에서 수행한 2차 확증(v4, 사전 등록 docs/37)의 결과다. 5.6절은 계산 비용, 5.7절은 예비 연구(v2)의 결과를 요약한다.

확증 근거는 5.2절과 5.5절뿐이다. 5.3절과 5.4절은 v3 테스트 결과를 본 뒤의 분석과 선택이므로 탐색적 결과로 읽어야 한다. 판정 규칙은 3.7절(표 6)과 같으며, 점추정 기준 지표(K1~K3, K6, K7)는 점추정이 목표 이상이면 '달성'이다.

### 5.1 핵심 지표 요약

{{table:v4_kpi}}

표 22. 핵심 지표의 목표와 판정. v3 열은 1차 확증(테스트 시드 800000번대, 사전 등록 docs/34), v4 열은 2차 확증(처음 보는 새 테스트 세트 1000000번대, 사전 등록 docs/37)의 값이다. 두 열은 평가 세트가 다르므로, 같은 세트에서 v3·v4 설정을 직접 비교한 값은 5.5절(표 34의 H-v)에 따로 보고한다. K4·K5는 시드 대응 차이와 시드·에피소드 계층 부트스트랩 95% CI이다.

{{fig:fig_v4_kpi.png|그림 10. 비율형 핵심 지표의 v3(1차 확증)·v4(2차 확증, 새 테스트) 값과 목표(점선). K6은 언어 입력이 줄인 속도 추종 오차의 비율이며, v3에서는 음수(언어가 오차를 키움)였다. 차이형 지표 K4·K5는 표 22에 있다.}}

- **1차 확증(v3)에서는 일곱 지표가 모두 미달했다.** 성공률은 목표에 가까웠으나(K1 {{s:k:K1:value}}, 목표 0.80), 점수기 고유 기여(K4)는 CI 하한이 0에 닿지 못했고, AMR 비열등(K5)·언어 지시 준수(K6)·실영상 제동 시작 예측(K7)은 목표와 거리가 멀었다.
- **2차 확증(v4)에서는 K1·K3·K6을 달성했고, K2·K4·K5·K7은 미달했다.** 값은 K1 {{s:k4:K1:value}}, K3 {{s:k4:K3:value}}, K6 {{p:k4:K6:speed_error_reduction}}이다. K2({{s:k4:K2:value}})는 목표 0.90에 조금 못 미쳤고, K4({{s:k4:K4:diff}})는 사실상 0이었다.
- v4의 개선은 큐레이션(점수기·풀·선별 하이퍼파라미터)을 그대로 두고 하위 정책과 학습 블록만 바꾼 결과다(3.6절). 같은 새 테스트에서 v4 설정은 v3 설정보다 CARE 2% 성공률이 {{s:k4:comparisons:v4_vs_v3_care:diff}} 높았다(5.5절).

### 5.2 1차 확증(v3)

1차 확증은 사전 등록(docs/34, 커밋 `bee8b95`)에 따라 주행 테스트 147 에피소드(시드 800000번대), 반사실 언어 세트 63 에피소드, AMR 테스트 72 에피소드, 실영상 5-fold × 시드 2개로 수행했다. CARE의 $\lambda$=0.5, $\rho$=0.9(AMR은 $\lambda$=0, $\rho$=0.95)는 테스트 전에 검증 세트에서 정했다(4.2.3절).

**(1) 주행 폐루프(K1~K3).**

{{table:v3_driving}}

표 23. v3 주행 테스트 결과(평균 ± 시드 표준편차). 예산 2%는 시드 10개, 1%·5%와 전체 데이터는 시드 5개다. 혼합 방법은 같은 학습 시드에서 공유 저장소(예산의 $\rho$)를 공유하고 점수 몫만 다르다(3.4절). *는 GT 위험 라벨을 쓰는 특권 정보 조건이다.

- **K1(미달)**: CARE 2%의 성공률({{s:k:K1:value}} ± {{s:k:K1:sd}})은 목표 0.80에 미달했다.
- **K2(미달)**: 전체 데이터의 성공률({{s:k:K2:full}})이 높아 비율은 {{s:k:K2:value}}에 그쳤다(목표 0.90).
- **K3(미달)**: 위험 시나리오 성공률은 {{s:k:K3:value}} ± {{s:k:K3:sd}}에 그쳤다(목표 0.75).
- 감속 트리거 단독 선택은 예비 연구(5.7절)와 같은 위험 편향 붕괴를 보였다. 성공률이 0.1 아래였고(표 23), 속도 오차가 커 진행 기준을 넘지 못했다. CARE − 감속 트리거 단독의 차이: {{s:k:comparisons:care_vs_trigger_only_b0.02:diff}}.

{{fig:fig_v3_budget.png|그림 11. v3 주행 테스트의 예산별 폐루프 성공률(CARE, 무작위(공유 저장소); 오차 막대는 시드 표준편차). 점선은 전체 데이터(100%), 붉은 파선은 K1 목표(0.80)다. 예산 1%·5%는 사전 등록상 보조(탐색적) 분석이다.}}

**(2) 점수기 고유 기여(K4)와 보조 가설.** 표 24는 사전 등록 가설 검정이다.

| 비교(예산 2%, 시드 10개 대응) | 차이 | 계층 부트스트랩 95% CI | 단측 $p$ | Holm 보정 $p$ | 판정 |
|---|---|---|---|---|---|
| K4: CARE − 저장소 + 감속 트리거 | {{s:k:K4:diff}} | [{{s:k:K4:lo}}, {{s:k:K4:hi}}] | {{s:k:K4:p_le0:3}} | {{s:k:K4:p_holm:3}} | 미달 |
| H-a: CARE − 무작위(공유 저장소) | {{s:k:comparisons:care_vs_random_shared_b0.02:diff}} | [{{s:k:comparisons:care_vs_random_shared_b0.02:lo}}, {{s:k:comparisons:care_vs_random_shared_b0.02:hi}}] | {{s:k:comparisons:care_vs_random_shared_b0.02:p_le0:4}} | {{s:k:comparisons:care_vs_random_shared_b0.02:p_holm:3}} | 지지 |
| H-b: CARE − 저장소 + 오라클* | {{s:k:comparisons:care_vs_mix_oracle_b0.02:diff}} | [{{s:k:comparisons:care_vs_mix_oracle_b0.02:lo}}, {{s:k:comparisons:care_vs_mix_oracle_b0.02:hi}}] | {{s:k:comparisons:care_vs_mix_oracle_b0.02:p_le0:3}} | {{s:k:comparisons:care_vs_mix_oracle_b0.02:p_holm:3}} | 차이 판별 안 됨 |
| (탐색) 저장소 + 감속 트리거 − 무작위 | {{s:k:comparisons:mix_trigger_vs_random_shared_b0.02:diff}} | [{{s:k:comparisons:mix_trigger_vs_random_shared_b0.02:lo}}, {{s:k:comparisons:mix_trigger_vs_random_shared_b0.02:hi}}] | {{s:k:comparisons:mix_trigger_vs_random_shared_b0.02:p_le0:3}} | 해당 없음 | - |
| (탐색) CARE − 무작위, 예산 1%(시드 5개) | {{s:k:comparisons:care_vs_random_shared_b0.01:diff}} | [{{s:k:comparisons:care_vs_random_shared_b0.01:lo}}, {{s:k:comparisons:care_vs_random_shared_b0.01:hi}}] | {{s:k:comparisons:care_vs_random_shared_b0.01:p_le0:4}} | 해당 없음 | - |
| (탐색) CARE − 무작위, 예산 5%(시드 5개) | {{s:k:comparisons:care_vs_random_shared_b0.05:diff}} | [{{s:k:comparisons:care_vs_random_shared_b0.05:lo}}, {{s:k:comparisons:care_vs_random_shared_b0.05:hi}}] | {{s:k:comparisons:care_vs_random_shared_b0.05:p_le0:4}} | 해당 없음 | - |

표 24. v3 사전 등록 가설 검정(docs/34 3절). 신뢰구간은 시드·에피소드 2단계 계층 부트스트랩(10,000회)이고, 단측 $p$는 차이 ≤ 0인 재표집의 비율이다. Holm 보정 군은 K4·H-a·H-b다. '(탐색)' 행은 사전 등록에서 보조 분석으로 정한 기술 통계이며 다중 비교 보정을 하지 않았다.

- **K4(미달)**: CARE는 같은 저장소에 감속 트리거 점수 몫을 더한 통제군보다 {{s:k:K4:diff}} 높았으나, 95% CI 하한이 0 아래({{s:k:K4:lo}})였고 Holm 보정 $p$도 0.05를 넘었다({{s:k:K4:p_holm:3}}). 시드 대응 $t$ 구간([{{s:k:K4:seed_t:lo}}, {{s:k:K4:seed_t:hi}}])은 0을 포함하지 않았지만, 사전 등록 판정은 계층 부트스트랩으로 한다.
- **H-a(지지)**: 공유 저장소를 고정하고 점수 몫만 무작위 몫과 바꾼 비교에서 CARE가 {{s:k:comparisons:care_vs_random_shared_b0.02:diff}} 높았다. 저장소 클립이 같으므로 이 차이는 예산의 10%인 점수 몫의 내용에서 나온다.
- 탐색 비교에서 예산 1%·5%의 CARE − 무작위도 양(+)이었고 CI가 0을 포함하지 않았다(그림 11). 감속 트리거 점수 몫의 무작위 대비 차이는 CI가 0을 포함했다.

**(3) 시나리오와 선택 구성(보조, 탐색적).**

{{table:v3_driving_scenario}}

표 25. v3 시나리오별 성공률(예산 2%는 시드 10개, 전체는 시드 5개 평균). 칸별 신뢰구간은 계산하지 않았으며 탐색적 기술 통계다.

- CARE가 무작위보다 높았던 칸은 주로 위험 시나리오(선행차 급제동, 정체)와 밀집이었다. 추종에서는 무작위가 더 높았다(표 25).
- 보행자 횡단은 모든 2% 방법에서 0.5 아래였고, 전체 데이터에서도 다른 시나리오보다 낮았다. 이 시나리오의 낮은 성공률은 v4 진단(5.3절)의 근거가 되었다.

{{table:v3_selection}}

표 26. v3 예산 2% 선택 데이터의 구성(시드 평균). 위험 프레임 비중은 선택 프레임 가운데 GT 위험 프레임의 비율, 위험 이벤트 회수율은 풀의 위험 이벤트 가운데 선택에 포함된 비율, 에피소드 포괄률은 선택 클립이 나온 에피소드의 비율이다.

- 혼합 방법(CARE·감속 트리거·오라클)의 선택 전체 위험 비중은 0.12~0.15로 무작위(0.07)의 1.6~2배였고, 감속 트리거 단독은 0.6을 넘었다.
- 세 혼합 방법은 위험 비중·회수율·엔트로피·에피소드 포괄률이 서로 비슷했다. 이 구성 차이만으로는 K4의 차이를 설명하기 어렵다(5.3절 (4)).

**(4) 언어 지시 준수(K6).**

{{table:v3_language_cf}}

표 27. v3 반사실 언어 평가(시드 5개). 같은 (시나리오, 시드)를 세 스타일 지시문으로 각각 평가했다. '신중−민첩 추종 간격 차'는 같은 (시나리오, 시드)에서 신중 지시와 민첩 지시의 평균 추종 간격 차이다.

- **K6(미달)**: CARE 2%에서 언어를 넣으면 속도 추종 오차가 {{s:k:K6:nolang:speed_error:2}} → {{s:k:K6:lang:speed_error:2}} m/s로 오히려 늘었다(감소율 {{p:k:K6:speed_error_reduction}}).
- 전체 데이터(보조)에서는 {{s:k:K6:by_data:full:nolang:speed_error:2}} → {{s:k:K6:by_data:full:lang:speed_error:2}} m/s로 {{p:k:K6:by_data:full:speed_error_reduction}} 줄었다.
- 스타일별 추종 간격 분리도는 두 데이터 모두 언어를 넣으면 커졌다(CARE {{s:k:K6:nolang:style_sep:2}} → {{s:k:K6:lang:style_sep:2}} s). 즉 2% 데이터의 정책은 지시문으로 추종 간격은 구분했지만 목표 속도는 따르지 못했다.

**(5) AMR(K5).**

{{table:v3_robot}}

표 28. v3 AMR 테스트 결과(72 에피소드; 2%는 시드 5개, 전체는 시드 3개). AMR 튜닝(검증, 시드 0)은 $\lambda$=0, $\rho$=0.95를 골랐다.

- **K5(미달)**: CARE − 무작위(공유 저장소)는 {{s:k:K5:diff}} [95% CI {{s:k:K5:lo}}, {{s:k:K5:hi}}]로, CI 하한이 비열등 한계 −0.03보다 훨씬 낮았다.
- 전체 데이터 정책의 성공률({{s:k:robot_full_success}})도 전문가({{s:k:robot_expert_success}})의 절반 수준이었다. AMR 과제는 데이터 선택 이전에 이 정책에게 어려웠다.

**(6) 실영상(K7).**

{{table:v3_comma}}

표 29. v3 comma.ai 실영상 개루프 결과(5-fold 블록 교차검증 × 시드 2개 = 10회 실행, ± 는 실행 간 표준편차). '특징 토큰'은 실제 YOLOv8n 검출 특징 입력 여부다. 제동 시작 AUROC는 현재 제동 중이 아닌 프레임에서 1초 안의 제동 시작을 판별한 값이다(3.7절).

- **K7(미달)**: 전체 데이터 특징 토큰 정책의 제동 시작 AUROC는 {{s:k:K7:value}} ± {{s:k:K7:sd}}에 그쳤다(특징 토큰 없음 {{s:k:K7:nofeat}}). 예비 연구와 마찬가지로 우연 수준이었다.
- 예산 10·20%의 방법 간 AUROC도 0.51~0.58 범위였다. 감속 트리거 단독은 제동 구간 MAE가 가장 낮았지만 전체 MAE가 가장 높아, 예비 연구의 개루프 트레이드오프를 재현했다.

### 5.3 미달 원인 진단(v3 원자료, 사후 분석)

이 절의 수치는 v3 결과를 본 뒤 원자료를 다시 분석한 것이다(docs/36 1절, 2.0~2.0d, 2.1). 진단은 v4 후보를 정하는 데만 썼고, 그 효과는 개발 세트(5.4절)와 새 테스트(5.5절)에서 따로 평가했다. 표 30에 진단과 대응을 정리했다.

| KPI | v3 원자료의 관찰 | 진단 가설 | v4 후보(3.6절) |
|---|---|---|---|
| K1·K3 | 보행자 횡단·정지-출발·선행차 급제동의 CARE 성공률이 낮다(표 25) | 2프레임($t$, $t-2$) 특징으로는 접근 속도·횡이동 같은 1초 단위 추세를 읽기 어렵다 | P2 시간 특징 인코더(H=8, GRU) |
| K2 | 전체 데이터 성공률이 높아져 비율이 낮아졌다 | 2% 데이터의 위험 클립을 더 효율적으로 학습해야 한다 | P3 위험 맥락 보조 헤드, P4 위험 가중 손실 |
| K5 | 전체 데이터 정책: 정지 작업자 시나리오 주행 중 충돌 27/36, 빈 통로 진행 미달 15/36. 위험 프레임 개루프 MAE가 전체 평균의 4배 | 위험 프레임(개루프 테스트의 4%)의 학습 비중이 작고, 저속 편향이 있다 | P4, 도메인별 개발 선택 |
| K6 | CARE 2%는 언어 유무와 관계없이 '보통'·'민첩' 지시에서 목표(약 16 m/s)보다 느렸다(평균 12~14 m/s) | 적은 데이터에서 FiLM이 지시문에 과적합하고, 숫자 토큰별 예시가 적어 목표 속도의 대응을 배우지 못한다 | T1 지시문 드롭아웃, P6 수치 목표 인코딩, P7 목표 속도 잔차 |
| K7 | 제동 시작 AUROC: 정책 0.528, CARE 점수기 0.518(fold별 0.41~0.64), 현재 자차 가속도 $-a_t$ 단독 0.802 | 정책의 고유 감각 입력이 현재 속도 하나뿐이다 | P5 자차 운동 이력 |
| K4 | (초기 진단) 선택 전체의 위험 비중이 CARE 0.146, 트리거 혼합 0.131로 비슷하다 | 점수 상위 클립이 같은 에피소드에 몰려 다양성이 떨어진다 | S1 점수 몫 에피소드 상한 → 측정 후 기각 |

표 30. v3 미달 원인 진단과 v4 후보. 수치 출처는 docs/36(1절, 2.0절 AMR, 2.0b절 실영상, 2.0c절 반사실, 2.1절 S1 측정)이며, 모두 v3 결과를 본 뒤의 사후 분석이다.

진단의 세부는 다음과 같다.

1. **AMR의 정지 작업자 충돌과 저속 편향(K5).** v3 전체 데이터 정책(시드 3개)의 실패는 두 양상에 몰려 있었다. 정지 작업자 시나리오에서는 36회 중 27회가 주행 중 충돌이었다. 천천히 다가가다 완전히 멈추지 못하고 밀려 들어가는 양상이었다(시드 0 평균 속도: 정책 0.67, 전문가 0.78 m/s). 빈 통로 시나리오에서는 36회 중 15회가 진행 미달이었고, 평균 속도가 0.94 m/s로 전문가 1.19 m/s보다 느렸다. 개루프 위험 프레임 MAE는 0.516으로 전체 평균 0.129의 4배였고, 위험 프레임은 개루프 테스트 프레임의 4%(546/13,500)였다.
2. **실영상 점수기와 단순 기준선(K7).** CARE 실영상 점수기의 제동 시작 AUROC는 0.518로 정책(0.528)과 비슷한 우연 수준이었다. 반면 정책 없이 현재 자차 가속도 $-a_t$ 하나만 점수로 쓰면 0.802였다. 영상·검출 특징보다 자차 운동 상태가 1초 안의 제동 시작과 더 강하게 연관되어 있었다.
3. **반사실 스타일별 저속 편향(K6).** v3 반사실 결과를 스타일별로 나누면, CARE 2% 정책은 언어 유무와 관계없이 '보통'·'민첩' 지시에서 목표 속도보다 느리게 달렸다. 전체 데이터 정책은 언어가 있으면 오차가 절반 가까이 줄었다(표 27). 이 차이는 데이터가 적을 때 지시문의 숫자 → 목표 속도 대응을 배우지 못한 것으로 해석했다. v4 개발 세트에서 P6(숫자 목표를 결정적으로 읽어 입력)을 더해도 반사실 오차가 줄지 않자(5.4절), 해석적 목표 속도 추종 기준 위에 잔차를 학습하는 P7을 추가했다.
4. **S1 측정 결과(K4).** 초기 진단은 선택 전체의 위험 비중(표 26)을 점수 몫의 위험 비중으로 잘못 읽은 것이었다. 점수 몫만 보면 CARE의 위험 비중은 약 0.80으로, 감속 트리거 혼합(0.64)보다 오히려 높았다. 또 CARE 2%의 점수 몫 36클립은 시드 0·1·2 모두에서 이미 서로 다른 36개 에피소드에서 나왔다. 따라서 에피소드당 1클립 상한(S1, $c$=1)은 CARE의 선택을 바꾸지 않고 트리거 혼합의 선택만 바꾼다(31 → 36 에피소드, 점수 몫 위험 비중 0.64 → 0.58). 이는 비교 기준선만 바꾸므로 S1은 개발 비교 전에 기각했다. v4에는 K4를 직접 겨냥한 개선이 없다.

### 5.4 v4 개발 세트 선택(적응적 선택, 확증 근거 아님)

v4 후보는 새 개발 세트(주행 84 에피소드, 시드 160000번대; AMR 72 에피소드, 시드 460000번대)에서 CARE 2%, 시드 3개로 비교했다. 채택 규칙은 '측정한 후보 가운데 개발 세트 CARE 2% 성공률(시드 3개 평균)이 가장 높은 것'이며, 후보별 세부 규칙은 각 후보의 결과가 나오기 전에 문서로 고정했다(4.7절, docs/36 3절·2.0c·2.0d). 주행과 AMR은 각자의 개발 세트로 따로 정했다.

{{table:v4_dev}}

표 31. v4 개발 세트 결과(CARE 2%, 시드 3개 평균 ± 시드 표준편차). '전부'는 P2+P3+T1+P4다. P5는 v3 실영상 결과 뒤, P6은 주행 '전부'의 반사실 결과 뒤, P7은 '전부+P6'의 결과 뒤에 추가한 후보다. T1은 별도 규칙(반사실 개발 세트의 언어 있음 속도 오차가 v3 이하)으로 판정했다.

{{table:v4_dev_lang}}

표 32. 주행 반사실 개발 세트(시드 170000번대, 시드 2개)의 속도 추종 오차. 'lang'은 언어 있음, 'nolang'은 같은 설정에서 언어 입력을 뺀 정책이다.

**채택.** 주행과 AMR 모두 '전부+P7'(P2+P3+T1+P4+P6+P7)이 채택되었다.

- **주행**: 개발 세트 성공률은 v3 기준 {{s:k4:dev_choice:driving:dev_success:v3}}에서 단일 블록 {{s:k4:dev_choice:driving:dev_success:p3}}~{{s:k4:dev_choice:driving:dev_success:t1}}, '전부' {{s:k4:dev_choice:driving:dev_success:all}}, '전부+P6' {{s:k4:dev_choice:driving:dev_success:all_p6}}, '전부+P7' {{s:k4:dev_choice:driving:dev_success:all_p7}}의 순서로 올랐다(P5 제외).
- **반사실 개발 세트(표 32)**: '전부'와 '전부+P6'에서는 언어를 넣은 정책의 오차({{s:k4:dev_choice:driving:devcf:all:lang:2}}, {{s:k4:dev_choice:driving:devcf:all_p6:lang:2}} m/s)가 언어를 뺀 정책({{s:k4:dev_choice:driving:devcf:all:nolang:2}} m/s)보다 컸다. 숫자 목표를 입력으로 주는 것(P6)만으로는 저속 편향이 남았다. '전부+P7'에서 처음으로 언어 있음({{s:k4:dev_choice:driving:devcf:all_p7:lang:2}} m/s)이 언어 없음보다 작아졌다.
- **AMR**: 단일 블록과 '전부'는 v3 기준({{s:k4:dev_choice:robot:dev_success:v3}})과 비슷했고('전부' {{s:k4:dev_choice:robot:dev_success:all}}), '전부+P7'에서 크게 올랐다({{s:k4:dev_choice:robot:dev_success:all_p7}}). AMR 개발 세트의 개선은 대부분 P7에서 나왔다.

**기각.**

- **S1(점수 몫 에피소드 상한)**: 개발 비교 전에 측정으로 기각했다(5.3절 (4)).
- **P5(자차 운동 이력)**: 주행 개발 세트 성공률({{s:k4:dev_choice:driving:dev_success:p5}})이 v3 기준보다 낮았고 시드 간 편차가 컸다(표 31). 자차의 최근 속도 변화를 그대로 따라 하는 관성 추종(copycat) 문제로 해석했다. AMR에서는 v3 기준과 비슷했다({{s:k4:dev_choice:robot:dev_success:p5}}). 이 기각 때문에 v4 실영상 정책에도 속도 이력이 없다(5.5절 (5)).

**해석상 주의.** 이 선택은 확증 근거가 아니다. 개발 세트는 여러 번 재사용되었고, 후보 집합 자체가 앞선 결과를 보고 늘어났다(P5·P6·P7). 최대 성공률 후보를 고르는 규칙은 개발 세트 값을 낙관적으로 만든다. 실제로 '전부+P7'의 주행 개발 세트 값({{s:k4:dev_choice:driving:dev_success:all_p7}})은 새 테스트 값({{s:k4:K1:value}}, 5.5절)보다 조금 높았다. 블록별 기여도 개발 세트의 시드 3개 비교일 뿐이며, 새 테스트에서 블록별 절제는 하지 않았다. 확증되는 것은 '채택된 v4 설정 전체'의 효과뿐이다(5.5절의 H-v).

### 5.5 2차 확증(v4, 새 테스트 세트)

2차 확증은 사전 등록(docs/37, 커밋 `6f97127`, 2026-10-04 23:26:29 UTC)을 커밋한 뒤, 처음 보는 새 테스트 세트(주행 1000000번대 147 에피소드, 반사실 1050000번대, AMR 1100000번대)에서 수행했다. 풀·점수기·풀 점수·선별 하이퍼파라미터는 v3와 같고, 정책·학습 설정만 5.4절의 채택 조합이다. 같은 새 테스트에서 v3 설정(CARE 2%·무작위 2%, 시드 10개)도 함께 평가해 개선 효과를 추정했다. 실영상에서는 아무것도 튜닝하지 않고 주행 조합을 그대로 적용했다.

**(1) 주행 폐루프(K1~K3).**

{{table:v4_driving}}

표 33. 새 테스트 세트의 주행 결과(평균 ± 시드 표준편차). '설정' 열의 v3는 v3 정책·학습 설정, v4는 채택 조합이다. 선별 방법·풀·점수기는 두 설정이 같다.

- **K1(달성)**: v4 CARE 2%의 성공률은 {{s:k4:K1:value}} ± {{s:k4:K1:sd}}(시드 {{s:k4:K1:n_seeds:0}}개)로 목표 0.80을 넘었다.
- **K2(미달)**: 전체 데이터 성공률({{s:k4:K2:full}})에 대한 비율은 {{s:k4:K2:value}}에 그쳤다. v3({{s:k:K2:value}})보다 높아졌으나 목표 0.90에 미달했다.
- **K3(달성)**: 위험 시나리오 성공률({{s:k4:K3:value}})이 목표 0.75를 넘었다. 다만 목표와의 차이가 0.01 미만이어서, 시드 변동을 고려하면 경계선의 달성이다. 사전 등록 판정은 점추정 기준이므로 '달성'으로 보고한다.
- 같은 새 테스트에서 v3 설정의 CARE 2% 성공률({{s:k4:driving:v3:care:success}})은 v3 자체 테스트의 값({{s:k:K1:value}})보다 낮았다. 새 테스트 세트가 v3 테스트보다 어려웠을 가능성이 있으며, v3·v4의 절대값 비교는 같은 세트 안에서만 해석한다.

**(2) 가설 검정(K4, H-a, H-v).**

| 비교(예산 2%, 시드 10개 대응) | 차이 | 계층 부트스트랩 95% CI | 시드 대응 $t$ 95% CI | 단측 $p$ | Holm 보정 $p$ | 판정 |
|---|---|---|---|---|---|---|
| K4: v4 CARE − v4 저장소 + 감속 트리거 | {{s:k4:K4:diff}} | [{{s:k4:K4:lo}}, {{s:k4:K4:hi}}] | [{{s:k4:K4:seed_t:lo}}, {{s:k4:K4:seed_t:hi}}] | {{s:k4:K4:p_le0:3}} | {{s:k4:K4:p_holm:3}} | 미달 |
| H-a: v4 CARE − v4 무작위(공유 저장소) | {{s:k4:comparisons:care_vs_random_shared:diff}} | [{{s:k4:comparisons:care_vs_random_shared:lo}}, {{s:k4:comparisons:care_vs_random_shared:hi}}] | [{{s:k4:comparisons:care_vs_random_shared:seed_t:lo}}, {{s:k4:comparisons:care_vs_random_shared:seed_t:hi}}] | {{s:k4:comparisons:care_vs_random_shared:p_le0:4}} | {{s:k4:comparisons:care_vs_random_shared:p_holm:3}} | 지지 |
| H-v: v4 CARE − v3 CARE | {{s:k4:comparisons:v4_vs_v3_care:diff}} | [{{s:k4:comparisons:v4_vs_v3_care:lo}}, {{s:k4:comparisons:v4_vs_v3_care:hi}}] | [{{s:k4:comparisons:v4_vs_v3_care:seed_t:lo}}, {{s:k4:comparisons:v4_vs_v3_care:seed_t:hi}}] | < 0.0001 | < 0.001 | 지지 |
| (탐색) v4 무작위 − v3 무작위 | {{s:k4:comparisons:v4_vs_v3_random_shared:diff}} | [{{s:k4:comparisons:v4_vs_v3_random_shared:lo}}, {{s:k4:comparisons:v4_vs_v3_random_shared:hi}}] | [{{s:k4:comparisons:v4_vs_v3_random_shared:seed_t:lo}}, {{s:k4:comparisons:v4_vs_v3_random_shared:seed_t:hi}}] | < 0.0001 | 해당 없음 | - |
| (탐색) v3 CARE − v3 무작위(새 테스트) | {{s:k4:comparisons:v3_care_vs_random_shared:diff}} | [{{s:k4:comparisons:v3_care_vs_random_shared:lo}}, {{s:k4:comparisons:v3_care_vs_random_shared:hi}}] | [{{s:k4:comparisons:v3_care_vs_random_shared:seed_t:lo}}, {{s:k4:comparisons:v3_care_vs_random_shared:seed_t:hi}}] | {{s:k4:comparisons:v3_care_vs_random_shared:p_le0:4}} | 해당 없음 | - |

표 34. v4 사전 등록 가설 검정(docs/37 3절, 새 테스트 세트). Holm 보정 군은 K4·H-a·H-v다. H-v와 v4 무작위 − v3 무작위는 계층 부트스트랩 10,000회 가운데 차이 ≤ 0인 재표집이 없었다. '(탐색)' 행은 사전 등록의 보조 분석이며 보정하지 않았다.

- **K4(미달)**: v4 CARE와 v4 트리거 혼합의 차이는 {{s:k4:K4:diff}} [95% CI {{s:k4:K4:lo}}, {{s:k4:K4:hi}}]이고, 구간이 0을 포함했다(Holm 보정 $p$={{s:k4:K4:p_holm:3}}). 두 방법의 성공률(각각 {{s:k4:driving:v4:care:success}}, {{s:k4:driving:v4:mix_trigger:success}})은 거의 같았다. v3(5.2절)와 달리 이번에는 시드 대응 $t$ 구간도 0을 넓게 포함했다. 동시에 계층 CI의 상한({{s:k4:K4:hi}})을 보면, 이 설정에서 점수기 고유의 선택 효과가 있더라도 약 0.03보다 클 가능성은 낮다.
- **H-a(지지)**: 공유 저장소를 고정한 무작위 대비 우위는 v4에서도 유지되었다({{s:k4:comparisons:care_vs_random_shared:diff}}, Holm 보정 $p$={{s:k4:comparisons:care_vs_random_shared:p_holm:3}}). 기술 통계로는 트리거 혼합({{s:k4:driving:v4:mix_trigger:success}})도 무작위({{s:k4:driving:v4:random_shared:success}})보다 높았다(검정하지 않음).
- **H-v(지지)**: 같은 새 테스트에서 v4 설정은 v3 설정보다 CARE 2% 성공률이 {{s:k4:comparisons:v4_vs_v3_care:diff}} 높았다. 무작위 2%에서도 개선 폭이 비슷했다({{s:k4:comparisons:v4_vs_v3_random_shared:diff}}, 탐색). 정책·학습 블록의 개선은 선별 방법과 거의 무관하게 더해졌다.
- **v3 효과의 재현(탐색)**: 같은 새 테스트에서 v3 설정의 CARE − 무작위({{s:k4:comparisons:v3_care_vs_random_shared:diff}})는 v3 테스트의 H-a({{s:k:comparisons:care_vs_random_shared_b0.02:diff}})와 방향·크기가 비슷했다. 다만 계층 CI 하한이 0에 걸쳤고({{s:k4:comparisons:v3_care_vs_random_shared:lo}}), 시드 대응 $t$ 구간은 0을 포함하지 않았다.
- 충돌률은 CARE가 무작위보다 낮았다(v4 CARE {{s:k4:driving:v4:care:collision}}, 트리거 혼합 {{s:k4:driving:v4:mix_trigger:collision}}, 무작위 {{s:k4:driving:v4:random_shared:collision}}, 표 33).

**(3) 언어 지시 준수(K6).**

{{table:v4_language_cf}}

표 35. 새 반사실 세트(시드 1050000번대, 63 에피소드)의 언어 평가(v4 설정, 시드 5개). 언어를 빼면 지시문에서 목표 속도를 읽는 P6·P7 경로도 함께 꺼진다.

- **K6(달성)**: v4 CARE 2%에서 언어를 넣으면 속도 추종 오차가 {{s:k4:K6:nolang:speed_error:2}} → {{s:k4:K6:lang:speed_error:2}} m/s로 {{p:k4:K6:speed_error_reduction}} 줄었다(목표 30%). 전체 데이터(보조)에서는 {{s:k4:K6:by_data:full:nolang:speed_error:2}} → {{s:k4:K6:by_data:full:lang:speed_error:2}} m/s로 {{p:k4:K6:by_data:full:speed_error_reduction}} 줄었다.
- 신중−민첩 추종 간격 차는 언어를 넣으면 CARE {{s:k4:K6:nolang:style_sep:2}} → {{s:k4:K6:lang:style_sep:2}} s, 전체 데이터 {{s:k4:K6:by_data:full:nolang:style_sep:2}} → {{s:k4:K6:by_data:full:lang:style_sep:2}} s로 커졌다.
- **해석 주의**: v4에는 P7(지시문의 목표 속도로 만든 비례 제어 기준 $a_0$ + 학습 잔차)이 들어 있다. 이 감소의 일부는 학습 파라미터가 없는 해석적 제어기 덕분이다. 같은 측정을 P7 없는 '전부+P6'으로 하면, 개발 세트에서는 언어가 오차를 오히려 키웠다(표 32). 새 테스트에서는 '전부+P6'을 평가하지 않았다. 해석은 6.5절에서 다룬다.

**(4) AMR(K5).**

{{table:v4_robot}}

표 36. 새 AMR 테스트 세트(시드 1100000번대)의 결과(2%는 시드 5개, 전체는 시드 3개). 선별 하이퍼파라미터는 v3 AMR 튜닝값($\lambda$=0, $\rho$=0.95)이다.

- **K5(미달)**: v4 CARE − v4 무작위는 {{s:k4:K5:diff}} [95% CI {{s:k4:K5:lo}}, {{s:k4:K5:hi}}]였다. 점추정은 v3({{s:k:K5:diff}})보다 0에 가까웠지만, CI 하한이 비열등 한계 −0.03보다 낮아 비열등을 판정하지 못했다.
- 절대 성공률은 크게 올랐다. CARE {{s:k:K5:care}} → {{s:k4:K5:care}}, 무작위 {{s:k:K5:random_shared}} → {{s:k4:K5:random_shared}}, 전체 데이터 {{s:k:robot_full_success}} → {{s:k4:robot_full_success}}(v3 → v4). 두 값의 테스트 시드는 서로 다르므로, 이 비교는 같은 세트의 대응 비교가 아니다.
- 트리거 혼합({{s:k4:robot:v4:mix_trigger:success}})과 무작위의 성공률은 같았다.

**(5) 실영상(K7).**

{{table:v4_comma}}

표 37. v4 설정의 comma.ai 실영상 개루프 결과(v3와 같은 5-fold × 시드 2개, ± 는 10회 실행의 표준편차). 실영상에서는 아무것도 튜닝하지 않고 주행 개발 조합(P7 이득은 주행 값)을 그대로 적용했다. GT 위험 라벨이 없어 오라클은 없고, 감속 트리거 단독은 v4에서 평가하지 않았다.

- **K7(미달)**: v4 전체 데이터 정책의 제동 시작 AUROC는 {{s:k4:K7:value}} ± {{s:k4:K7:sd}}에 그쳤다(v3 {{s:k:K7:value}}). fold·시드별 값은 {{s:k4:K7:min:2}}~{{s:k4:K7:max:2}} 범위에 흩어졌다.
- 같은 시험 프레임에서 정책 없이 현재 자차 가속도 $-a_t$만 쓴 기준선의 AUROC는 {{s:k4:K7:baseline_neg_accel}}에 이르렀다. v4 정책은 이 단순 기준선에 크게 못 미쳤다. P5가 개발 세트에서 기각되어(5.4절) v4 정책에도 속도 이력 입력이 없다.
- 예산 10·20%의 방법 간 AUROC는 0.52~0.57 범위로, 방법 간 차이를 가릴 분해능이 없었다.

### 5.6 계산 비용

표 38은 큐레이션과 하위 정책의 계산 비용이다. 점수기는 예비 연구·v3·v4에서 같으므로 예비 연구의 측정값을 그대로 쓴다.

| 항목 | 값 | 비고 |
|---|---|---|
| 맥락 점수기 파라미터 | {{s:cost_scorer_params:0}} | 멀티채널 CNN-GRU, 창 16. v3·v4에서 재학습하지 않음 |
| 점수기 1회 추론(창 1개, 배치 1) | {{s:cost_scorer_onnx_ms:3}} ms(ONNX Runtime) / {{s:cost_scorer_torch_ms:2}} ms(PyTorch) | x86 CPU 1스레드 |
| 검출기 YOLOv8n(참고) | {{s:cost_yolo640_ms:1}} ms(640) / {{s:cost_yolo320_ms:1}} ms(320) | x86 CPU 4스레드 |
| 하위 정책 파라미터 | v3 {{s:k4:cost:v3:n_params:0}} / v4 {{s:k4:cost:v4:n_params:0}} | v4 증가분은 GRU 특징 인코더, 보조 헤드, P6 MLP와 헤드 입력 확장이다. P7은 학습 파라미터가 없고, 보조 헤드는 추론에 쓰지 않는다 |
| 정책 학습 처리량(통제 측정) | v3 1,657 / 1,579, v4 1,466 / 1,433 samples/s | docs/31 11.4절. 1스레드, 배치 128, 같은 기계에서 교대 2회 측정. v4는 P6·P7을 넣기 전 조합(564,526 파라미터) |
| 정책 학습 처리량(본 실험 기록) | v3 {{s:k4:cost:v3:samples_per_s:0}}, v4 {{s:k4:cost:v4:samples_per_s:0}} samples/s(중앙값) | 새 테스트 2% 학습의 기록. 작업자 4개가 CPU를 공유한 상태라 참고치다 |

표 38. 계산 비용(x86 클라우드 CPU). 배포 대상 임베디드 장치(Jetson Orin Nano)에서의 지연·전력은 측정하지 않았다.

- 통제 측정에서 v4 정책 학습은 v3보다 약 10% 느렸다. 특징 이력 조회, GRU의 8단계 순차 계산, 보조 헤드 역전파가 원인이다.
- 수집 시점 큐레이션의 비용(점수기)은 v3·v4에서 바뀌지 않았다. v4의 추가 비용은 모두 학습·추론하는 정책 쪽에 있다.

### 5.7 예비 연구(v2) 결과 요약

예비 연구는 v3 이전의 벤치마크와 정책(특징 토큰 없음)으로 수행했다. 설계와 결함 진단은 4.2.1절에 있으며, 여기서는 본 연구의 판단에 쓰인 결과만 요약한다.

- **위험 편향 붕괴**: 위험 표본을 최대화하는 선별(감속 트리거, 맥락 이벤트, 오라클, 정책 손실 기반 오프라인 선별)은 주행 예산 2%에서 성공률 {{s:risk_methods_max_success_b0.02:2}} 이하로 무너졌다. 이 결과는 1~10% 예산과 AMR에서도 재현되었고, v3의 감속 트리거 단독(5.2절)에서도 다시 나타났다.
- **CARE − 무작위**: 시드 3개 탐색 실험에서는 신뢰구간이 0을 포함했으나({{s:hb:ours_vs_random_b0.02:diff}} [95% CI {{s:hb:ours_vs_random_b0.02:lo}}, {{s:hb:ours_vs_random_b0.02:hi}}]), 새 시드 7개의 사전 등록 확증 실험(docs/32)에서는 예산 2%의 우위가 지지되었다({{s:ct:H1_ours_vs_random_b0.02:diff}} [95% CI {{s:ct:H1_ours_vs_random_b0.02:lo}}, {{s:ct:H1_ours_vs_random_b0.02:hi}}]). 예산 1%의 우위와 혼합 통제군 대비 우위는 유의하지 않았다(Holm 보정 $p$ ≥ {{s:ct:H3a_ours_vs_mix_trigger:p_holm:2}}).
- **개루프와 폐루프**: 전체 구간 개루프 MAE는 폐루프 성공률과 강하게 상관했지만($r_s$={{s:ol_cl_spearman_mae:2}}), 예산 2% 안에서 위험 구간 MAE는 성공률과 오히려 양(+)의 상관을 보였다($r_s$={{s:ol_cl_within2_mae_hazard:2}}, $p$={{s:ol_cl_within2_mae_hazard_p:3}}).
- **판단하지 못한 것**: AMR에서 CARE({{s:robot:ours:0.02:success}})와 무작위({{s:robot:random:0.02:success}})의 차이는 유의하지 않았다. 실영상 제동 시작 AUROC는 전체 데이터에서도 {{s:comma:full:1.00:brake_onset_auroc:2}}에 그쳤다. 언어 효과는 지시 정보가 초기 속도로 누출되어 측정하지 못했다.

이 결과가 K1~K7의 목표값과 v3의 벤치마크 수정(B1~B3), 공유 저장소 선별(M1)의 근거가 되었다(3.7절, 4.2.2절).

## 5. 실험 결과

5.1~5.10절은 사전 등록 이전에 수행한 본 실험의 결과이며(폐루프는 시드 3개, 실영상은 fold 5개 × 시드 2개), 4.3절에 따라 모두 탐색적 결과로 보고한다. 5.1절의 점수기 진단과 5.10절의 비용 측정은 정책 학습 시드와 무관하다. 사전 등록 확증 실험의 결과는 5.11절에 따로 보고한다.

### 5.1 점수기와 선택 데이터의 구성

초기 라벨 셋으로 학습한 점수기를 새 큐레이션 풀(지시문 스타일 추가, 시드 분리)에 적용했다. 위험 프레임 판별 AUROC({{s:pool_scorer_auroc}})는 같은 프레임에서 역 TTC 특징 단독({{s:pool_ittc_auroc}})과 자차 감속도({{s:pool_action_auroc}})보다 높았다. 점수기는 물리 단서 하나보다 시간 맥락을 통합한 쪽이 위험 프레임을 더 잘 가려냈다.

**점수기는 자차 감속보다 먼저 반응하는가.** 1차 원고는 단일 사례 그림을 근거로 점수기가 자차 감속 전에 위험 확률을 높인다고 주장했다. 이를 풀 전체에서 확인했다. 풀의 위험 사건 {{s:lead_n_events:0}}개마다 사건 시작 3초 전부터 사건 끝까지의 창을 잡았다. 그 안에서 '위험 확률 $e_t$가 처음 0.5를 넘은 시각'과 '자차 가속도가 처음 −1 m/s² 아래로 내려간 시각'을 찾고, 선행 시간을 (감속 시각 − 점수 시각)으로 정의했다. 양수이면 점수기가 먼저 반응한 것이다.

- 두 신호가 모두 나타난 사건은 {{s:lead_n_both:0}}개였다. 선행 시간의 중앙값은 {{s:lead_median_s:2}} s, 사분위 범위는 {{s:lead_q25_s:2}} ~ {{s:lead_q75_s:2}} s였다.
- 점수기가 먼저 반응한 사건의 비율은 {{p:lead_frac_score_first:0}}로 절반에 못 미쳤다.
- 점수만 반응한 사건은 {{s:lead_n_score_only:0}}개, 감속만 나타난 사건은 {{s:lead_n_decel_only:0}}개였다.

따라서 **점수기는 자차 감속보다 앞서지 않았고, 대체로 동시 수준으로 반응했다.** 1차 원고의 '예측적 성질' 주장은 철회한다. 풀의 자차 가속도는 반응 지연을 넣은 전문가 제어기의 것이므로, 사람 운전 자료에서 같은 결과가 나오는지는 확인하지 않았다.

{{fig:fig_vla_episode.png|그림 2. 큐레이션 풀의 선행차 급제동 에피소드 사례. 사례는 급제동 시나리오 에피소드 가운데 고정 시드로 무작위 추출했으며, 대표성을 주장하지 않는다. 위: 정책 관측(64×64 렌더 영상). 가운데: 점수기의 위험 확률 $e_t$, 불확실성 $u_t$, 역 TTC 특징, GT 위험 구간(붉은 영역), 자차 가속도(감속 트리거의 입력)와 전문가 명령(행동 라벨). 아래: 예산 2%에서 CARE($\rho$=0.9)의 실제 선택, 감속 트리거, 무작위가 이 에피소드에서 고른 클립(본 실험 시드 0의 선택을 재현). CARE의 선택은 점수 몫과 저장소 몫을 구분하지 않고 표시했다. 점수기와 자차 감속의 선후 관계는 이 한 사례가 아니라 풀 전체 위험 사건의 선행 시간 분포(본문)로 판단한다.}}

선택 데이터의 구성은 방법마다 크게 달랐다(그림 3). 아래에는 회수율과 선택 데이터 내 위험 비중(표 5)을 함께 적었다.

- 주 예산 2%의 GT 위험 프레임 회수율 / 위험 비중
  - 오라클: {{s:driving:oracle:0.02:hazard_frame_recall}} / {{s:driving:oracle:0.02:hazard_share:2}}
  - 맥락 이벤트: {{s:driving:event:0.02:hazard_frame_recall}} / {{s:driving:event:0.02:hazard_share:2}}
  - 감속 트리거: {{s:driving:action_trigger:0.02:hazard_frame_recall}} / {{s:driving:action_trigger:0.02:hazard_share:2}}
  - CARE: {{s:driving:ours:0.02:hazard_frame_recall}} / {{s:driving:ours:0.02:hazard_share:2}}
  - 무작위: {{s:driving:random:0.02:hazard_frame_recall}} / {{s:driving:random:0.02:hazard_share:2}}
- 무작위의 회수율은 예산 비율 수준이고, 위험 비중은 풀의 위험 비율 수준이다.
- 위험 지향 방법의 선택 데이터는 위험 프레임이 다수를 차지했다. 위험 비중은 무작위의 약 {{s:risk_share_ratio_min:0}}~{{s:risk_share_ratio_max:0}}배였다(감속 트리거·맥락 이벤트·오라클·오프라인 손실).
- CARE는 예산의 90%를 무작위 저장소로 쓰므로, 선택 데이터의 위험 비중이 무작위보다 조금 높은 수준에 그친다. 즉 CARE의 선택 데이터는 '무작위 + 약간의 위험 강화'다.

5.2~5.5절은 위험 프레임을 최대한 모으는 전략이 정책을 망치는 양상과, 약간의 위험 강화가 보인 효과를 차례로 살핀다.

{{fig:fig_vla_selection.png|그림 3. 예산 2%에서 각 방법이 고른 데이터의 GT 위험 프레임 회수율과 맥락 라벨 분포 엔트로피(정규화). 회수율은 풀 전체 위험 프레임 가운데 선택된 비율이며, 선택 데이터 내 위험 비중은 본문에 함께 적었다.}}

### 5.2 주 결과: 예산 2%의 폐루프 성능

표 13은 예산 2%(풀의 {{s:driving:random:0.02:n_train_frames:0}} 프레임)에서 학습한 VLA-lite 정책의 테스트 결과다(147 에피소드, 시드 3개).

{{table:vla_main_driving}}

표 13. 예산 2% 폐루프·개루프 결과(평균 ± 시드 표준편차). '엣지 가능'은 수집 시점·정책 비의존·상수 비용(C1~C3)을 모두 만족하는지 나타낸다. *는 수집 시점에 계산할 수 없는 참고 기준이다. 개루프 MAE는 별도 개루프 테스트 풀(150 에피소드, 3프레임 간격 표본 {{s:openloop_n_frames:0}}프레임, 그중 GT 위험 프레임 {{s:openloop_n_hazard:0}}개)의 전문가 시연에 대한 행동 청크 오차(m/s²)이고, 위험 MAE는 그중 GT 위험 프레임만의 오차다. 지표 정의는 표 5에 있다. 전체 데이터(100%) 행의 엣지 가능 열 '-'는 선별이 아니므로 해당 없음을 뜻한다.

**(1) CARE의 평균 성공률이 수집 시점 방법 가운데 가장 높았으나, 무작위와의 차이는 시드 수준에서 결정적이지 않았다.**

- CARE의 성공률({{s:driving:ours:0.02:success}} ± {{s:driving:ours:0.02:success_sd}})은 무작위({{s:driving:random:0.02:success}} ± {{s:driving:random:0.02:success_sd}})보다 높았다.
- 시드·에피소드 계층 부트스트랩으로 구한 차이의 신뢰구간은 0을 포함한다({{s:hb:ours_vs_random_b0.02:diff}} [95% CI {{s:hb:ours_vs_random_b0.02:lo}}, {{s:hb:ours_vs_random_b0.02:hi}}], 단측 $p$={{s:hb:ours_vs_random_b0.02:p_le0:3}}). 시드 대응 $t$ 구간([{{s:hbt:ours_vs_random_b0.02:lo}}, {{s:hbt:ours_vs_random_b0.02:hi}}])은 더 넓다.
- 참고로 에피소드만 재표집한 구간({{b:ours_vs_random_b0.02}})은 0을 포함하지 않는다. 그러나 이 구간은 평가 분산만 반영한다(4.3절).
- 시드별 차이(시드 0 / 1 / 2: {{s:sdiff:0.02:0:3}} / {{s:sdiff:0.02:1:3}} / {{s:sdiff:0.02:2:3}})를 보면, 한 시드에서는 CARE가 무작위보다 낮았다. 시드 간 편차가 큰 무작위의 평균이 성공률이 낮은 시드에 끌려 내려간 영향도 있다.
- 이 차이는 사전 등록 확증 실험(새 시드 7개, 새 평가 세트)에서 다시 검정했으며, 예산 2%의 우위가 지지되었다({{s:ct:H1_ours_vs_random_b0.02:diff}} [95% CI {{s:ct:H1_ours_vs_random_b0.02:lo}}, {{s:ct:H1_ours_vs_random_b0.02:hi}}], 5.11절).
- 같은 선별법·예산·시드 번호로 언어 입력만 빼고 다시 학습한 사실상의 반복 실험(5.7절)에서는 무작위가 오히려 높았다({{s:lang:ours:nolang:success}} 대 {{s:lang:random:nolang:success}}).
- 평균으로는 충돌률({{s:driving:ours:0.02:collision}} 대 {{s:driving:random:0.02:collision}}), 속도 오차({{s:driving:ours:0.02:speed_error:2}} 대 {{s:driving:random:0.02:speed_error:2}} m/s), 위험 시나리오 성공률({{s:driving:ours:0.02:hazard_success}} 대 {{s:driving:random:0.02:hazard_success}})이 모두 CARE 쪽이 나았다.
- 예산 2%에서 관찰된 시드 3개 표준편차는 CARE가 무작위보다 작았다. 그러나 1·5·10% 예산에서는 비슷하거나 CARE가 더 컸고(표 15), 시드 3개의 표준편차 비교는 검정력이 거의 없다. 따라서 이를 CARE의 일반적 성질로 해석하지 않는다.

**(2) 위험 표본만 늘리는 방법은 모두 실패했다.**

- 성공률
  - 감속 트리거: {{s:driving:action_trigger:0.02:success}}
  - 맥락 이벤트: {{s:driving:event:0.02:success}}
  - 오라클: {{s:driving:oracle:0.02:success}}
  - 오프라인 손실: {{s:driving:offline_loss:0.02:success}}
  - 불확실성: {{s:driving:uncertainty:0.02:success}}
- 무작위 대비 차이는 계층 부트스트랩에서도 모두 0을 포함하지 않았다(예: 감속 트리거 {{s:hb:action_trigger_vs_random_b0.02:diff}} [{{s:hb:action_trigger_vs_random_b0.02:lo}}, {{s:hb:action_trigger_vs_random_b0.02:hi}}], 오라클 {{s:hb:oracle_vs_random_b0.02:diff}} [{{s:hb:oracle_vs_random_b0.02:lo}}, {{s:hb:oracle_vs_random_b0.02:hi}}], 오프라인 손실 {{s:hb:offline_loss_vs_random_b0.02:diff}} [{{s:hb:offline_loss_vs_random_b0.02:lo}}, {{s:hb:offline_loss_vs_random_b0.02:hi}}]).
- 이들은 충돌이 거의 없었다(충돌률: 오라클 {{s:driving:oracle:0.02:collision}}, 감속 트리거 {{s:driving:action_trigger:0.02:collision}}).
- 대신 속도 오차가 9~14 m/s에 달해 전문가 진행 거리의 80%를 넘지 못했다.
- 즉, 위험 상황 위주로 학습한 정책은 "항상 감속한다"는 단순한 해로 수렴했다. 본 논문은 이를 **위험 편향 붕괴**(risk-bias collapse)라 부른다(5.3절).
- 정책 의존 오프라인 기준선(오프라인 손실)도 이 붕괴를 피하지 못했다. 손실 상위 프레임은 대부분 행동 변화가 큰 제동 구간이라(위험 비중 {{s:driving:offline_loss:0.02:hazard_share:2}}), 결국 위험 지향 선택과 같은 문제를 겪는다.

**(3) 커버리지 지향 방법도 무작위보다 낮았다.**

- Coreset({{s:driving:coreset:0.02:success}})과 역 TTC 규칙({{s:driving:rule_ittc:0.02:success}})은 무작위보다 낮았고, 계층 신뢰구간도 0을 포함하지 않았다(Coreset {{s:hb:coreset_vs_random_b0.02:diff}} [{{s:hb:coreset_vs_random_b0.02:lo}}, {{s:hb:coreset_vs_random_b0.02:hi}}]).
- Coreset은 특징 공간의 외곽(드문 장면)을 우선해, 결과적으로 정상 주행 표본을 줄인다(위험 비중 {{s:driving:coreset:0.02:hazard_share:2}}).
- 등간격 선택({{s:driving:uniform:0.02:success}})은 무작위와 유의한 차이를 보이지 않았다({{s:hb:uniform_vs_random_b0.02:diff}} [{{s:hb:uniform_vs_random_b0.02:lo}}, {{s:hb:uniform_vs_random_b0.02:hi}}]).
- 데이터 선택 연구에서 무작위가 강한 기준선이라는 보고[39]와 일치한다.

**(4) 시나리오 가중 민감도.** 테스트 세트는 7개 시나리오를 같은 수로 담으므로 위험 시나리오가 4/7로 과대표집되어 있다. 풀의 시나리오 빈도로 가중해 성공률을 다시 계산했다. 가중 성공률(CARE {{s:driving:ours:weighted_success}}, 무작위 {{s:driving:random:weighted_success}}, 등간격 {{s:driving:uniform:weighted_success}}, 전체 데이터 {{s:driving:full:weighted_success}})의 순서는 균등 가중과 같았다. 위험 지향 방법(감속 트리거 {{s:driving:action_trigger:weighted_success}}, 오라클 {{s:driving:oracle:weighted_success}})은 여전히 최하위였다. 다만 이 가중치는 시뮬레이터 풀의 생성 빈도이며, 실제 도로의 시나리오 빈도를 대표하지 않는다. 가중 성공률의 신뢰구간은 계산하지 않았다.

### 5.3 위험 편향 붕괴의 분석

{{table:vla_scenarios_driving}}

표 14. 시나리오별 폐루프 성공률(예산 2%, 시드 평균; 전체 데이터는 100%). 칸마다 에피소드 21개 × 시드 3개이며, 칸별 신뢰구간은 계산하지 않았다. 위험 시나리오는 cut_in, lead_brake, stop_and_go, vru_crossing이다.

시나리오별 결과(표 14)는 붕괴의 성격을 보여 준다.

- **위험 지향 방법**(감속 트리거·맥락 이벤트·오라클)은 모든 시나리오에서 성공률이 0.35 이하였다. 위험 시나리오에서는 충돌을 피했지만, 비위험 시나리오인 정상 주행(free_drive)과 추종(follow)에서도 서행해 진행 기준을 넘지 못했다.
- **무작위**는 비위험 시나리오(정상·추종·밀집)에서 0.78~0.87로 높았고, 위험 시나리오 가운데 끼어들기({{s:driving:random:scen:cut_in:2}})에서도 높았다. 반면 선행차 급제동({{s:driving:random:scen:lead_brake:2}}), 보행자 횡단({{s:driving:random:scen:vru_crossing:2}}), 정체({{s:driving:random:scen:stop_and_go:2}})에서는 충돌로 실패가 잦았다.
- **CARE**
  - 같은 세 위험 시나리오에서 성공률(각각 {{s:driving:ours:scen:lead_brake:2}}, {{s:driving:ours:scen:vru_crossing:2}}, {{s:driving:ours:scen:stop_and_go:2}})이 무작위보다 높았다.
  - 위험 시나리오인 끼어들기({{s:driving:ours:scen:cut_in:2}})와 비위험 시나리오인 추종·밀집에서도 무작위보다 같거나 높았다.
  - 자유주행에서만 성공률이 다소 낮았다({{s:driving:ours:scen:free_drive:2}} 대 {{s:driving:random:scen:free_drive:2}}).
  - CARE의 위험 시나리오 성공률({{s:driving:ours:0.02:hazard_success}})은 끼어들기를 포함한 네 위험 시나리오의 평균이다.
  - 평균으로 보면 CARE의 차이는 정상 주행을 거의 희생하지 않으면서 위험 시나리오 대응이 나아진 데서 나온다. 다만 칸마다 표본이 작으므로(시드 3개) 시나리오별 차이는 기술 통계로만 읽어야 한다.
- 보행자 횡단은 전체 데이터(100%)로 학습한 정책도 성공률이 {{s:driving:full:scen:vru_crossing:2}}에 그쳤다. 이 시나리오의 상한은 데이터 선택보다 64×64 관측에서 작은 보행자를 식별하는 정책의 시각 해상도에 묶여 있을 가능성이 있다.

이 결과는 데이터 품질을 분포 이동으로 설명한 선행 연구[25]의 관점과 맞닿아 있다. 위험 과표집은 위험 상태의 행동 오차를 줄이지만, 학습 상태 분포를 배치 분포에서 멀어지게 해 정상 상태에서의 공변량 이동[75]을 키운다.

### 5.4 예산 규모에 따른 변화

{{fig:fig_vla_budget_curves.png|그림 4. 저장 예산에 따른 폐루프 성공률(왼쪽), 위험 시나리오 성공률(가운데), 선택 데이터의 위험 프레임 회수율(오른쪽). 오차 막대는 시드 표준편차, 점선은 수집 시점 계산 불가 방법, 일점쇄선은 전체 데이터(100%). 회수율은 예산과 함께 커지지만, 위험 지향 방법의 선택 데이터 내 위험 비중은 예산이 커질수록 줄어든다(본문).}}

{{table:vla_budget_success}}

표 15. 예산별 폐루프 성공률(평균 ± 시드 표준편차). 1·5·10% 예산에서는 핵심 6개 방법만 실행했으며, '-'는 미실행 칸이다. 마지막 행의 전체 데이터(100%)는 예산 열과 무관한 참고값({{s:driving:full:1.00:success}} ± {{s:driving:full:1.00:success_sd}})이며, 1% 열에 놓인 것은 표 형식 때문이다.

위험 지향 방법의 붕괴는 모든 예산에서 나타났다(그림 4, 표 15). 감속 트리거, 맥락 이벤트, 오라클은 1%에서 10%까지 모든 예산에서 성공률 0.26 이하였고, 무작위 대비 차이의 계층 신뢰구간은 모든 예산에서 0을 포함하지 않았다.

그러나 이 방법들의 학습 분포가 예산과 무관하게 고정된 것은 아니다. 예산이 커지면 회수율은 커지지만, 선택 데이터 내 위험 비중은 오히려 줄었다. 위험 프레임의 총량이 한정되어 있어, 예산이 커질수록 위험 구간 주변의 비위험 프레임이 함께 들어오기 때문으로 보인다(오라클의 회수율은 10%에서 {{s:driving:oracle:0.10:hazard_frame_recall:2}}에 이른다). 이에 따라 성공률도 부분적으로 회복되었다.

- 감속 트리거: 위험 비중 {{s:driving:action_trigger:0.01:hazard_share:2}}(1%) → {{s:driving:action_trigger:0.10:hazard_share:2}}(10%), 성공률 {{s:driving:action_trigger:0.01:success:2}} → {{s:driving:action_trigger:0.10:success:2}}
- 맥락 이벤트: 위험 비중 {{s:driving:event:0.01:hazard_share:2}} → {{s:driving:event:0.10:hazard_share:2}}, 성공률 {{s:driving:event:0.01:success:2}} → {{s:driving:event:0.10:success:2}}
- 오라클: 위험 비중 {{s:driving:oracle:0.01:hazard_share:2}} → {{s:driving:oracle:0.10:hazard_share:2}}, 성공률 {{s:driving:oracle:0.01:success:2}} → {{s:driving:oracle:0.10:success:2}}
- 그래도 10%에서 무작위({{s:driving:random:0.10:success:2}})에 크게 못 미쳤다.

같은 예산 범위(1~10%)에서 CARE({{s:driving:ours:0.01:hazard_share:2}}~{{s:driving:ours:0.10:hazard_share:2}})와 무작위({{s:driving:random:0.10:hazard_share:2}}~{{s:driving:random:0.01:hazard_share:2}})의 위험 비중은 예산에 거의 무관했다.

무작위 대비 CARE의 차이는 다음과 같다(표 16).

| 예산 | 평균 차이 | 계층 부트스트랩 95% CI(주 분석) | 단측 $p$ | 시드 대응 $t$ 95% CI | 에피소드 재표집 차이와 CI(참고) |
|---|---|---|---|---|---|
| 1% | {{s:hb:ours_vs_random_b0.01:diff}} | [{{s:hb:ours_vs_random_b0.01:lo}}, {{s:hb:ours_vs_random_b0.01:hi}}] | {{s:hb:ours_vs_random_b0.01:p_le0:3}} | [{{s:hbt:ours_vs_random_b0.01:lo}}, {{s:hbt:ours_vs_random_b0.01:hi}}] | {{b:ours_vs_random_b0.01}} |
| 2% | {{s:hb:ours_vs_random_b0.02:diff}} | [{{s:hb:ours_vs_random_b0.02:lo}}, {{s:hb:ours_vs_random_b0.02:hi}}] | {{s:hb:ours_vs_random_b0.02:p_le0:3}} | [{{s:hbt:ours_vs_random_b0.02:lo}}, {{s:hbt:ours_vs_random_b0.02:hi}}] | {{b:ours_vs_random_b0.02}} |
| 5% | {{s:hb:ours_vs_random_b0.05:diff}} | [{{s:hb:ours_vs_random_b0.05:lo}}, {{s:hb:ours_vs_random_b0.05:hi}}] | {{s:hb:ours_vs_random_b0.05:p_le0:3}} | [{{s:hbt:ours_vs_random_b0.05:lo}}, {{s:hbt:ours_vs_random_b0.05:hi}}] | {{b:ours_vs_random_b0.05}} |
| 10% | {{s:hb:ours_vs_random_b0.10:diff}} | [{{s:hb:ours_vs_random_b0.10:lo}}, {{s:hb:ours_vs_random_b0.10:hi}}] | {{s:hb:ours_vs_random_b0.10:p_le0:3}} | [{{s:hbt:ours_vs_random_b0.10:lo}}, {{s:hbt:ours_vs_random_b0.10:hi}}] | {{b:ours_vs_random_b0.10}} |

표 16. 예산별 CARE − 무작위 성공률 차이(시드 3개). 단측 $p$는 계층 부트스트랩에서 차이 ≤ 0인 재표집의 비율이다. 다중 비교 보정은 하지 않았다.

- 1%와 2%에서는 평균 차이가 양(+)이었으나, 계층 신뢰구간은 모두 0을 포함했다. 에피소드 재표집 기준으로만 0을 벗어났다.
- 5%와 10%에서는 차이가 0에 가깝거나 음(−)이었다.
- 1~2% 예산은 파일럿을 본 뒤 추가되었으므로(표 6), 이 패턴 자체도 탐색적이다.

예산이 5% 이상이면 무작위 선택도 위험 프레임을 수천 개 담게 되어, 위험 강화의 추가 이득이 없어진다는 해석이 가능하다. 이 해석이 맞다면 CARE는 **위험 표본이 절대적으로 부족한 영역**(소예산, 또는 위험이 더 희소한 도메인)에서만 의미가 있다. 그러나 본 실험은 이 해석을 직접 검정하지 않았다. 또한 $\rho$를 2% 검증에서 한 번 고른 값으로 고정했으므로, 예산이 클 때 $\rho$를 1에 더 가깝게 조정하면 결과가 달라질 수 있다.

주 예산 2%에서 CARE는 등간격 선택보다 평균이 높았지만, 계층 신뢰구간은 0을 포함했다({{s:hb:ours_vs_uniform_b0.02:diff}} [{{s:hb:ours_vs_uniform_b0.02:lo}}, {{s:hb:ours_vs_uniform_b0.02:hi}}]). 전체 데이터(100%)보다는 낮았다(에피소드 재표집 기준 {{b:ours_vs_full_b0.02}}). 2%의 데이터로 전체 데이터 성공률의 약 {{pp:ratio_ours_full_b0.02:0}}%를 얻은 셈이다.

### 5.5 혼합 비율과 불확실성 항의 절제

{{fig:fig_vla_mixing.png|그림 5. 예산 2%에서 저장소 비율 ρ에 따른 폐루프 성공률과 충돌률(테스트, 시드 3개). ρ=0은 점수 상위만, ρ=1은 무작위다. ρ와 λ는 검증 시나리오에서 이미 선택했으며(4.2절), 이 그림은 분석 목적의 사후 보고다.}}

{{table:vla_ablation}}

표 17. CARE 구성 요소 절제(예산 2%, 테스트, 시드 3개). 사후 분석이며, 다중 비교 보정은 하지 않았다.

표 17과 그림 5는 두 구성 요소의 기여를 보여 준다.

- **저장소 비율 $\rho$**: $\lambda$=0.5에서 $\rho$를 0.25 → 0.5 → 0.75 → 0.9로 높였을 때의 성공률(차례로 {{s:abl:0.5:0.25:success}}, {{s:abl:0.5:0.50:success}}, {{s:abl:0.5:0.75:success}}, {{s:abl:0.5:0.90:success}})은 0.25와 0.5에서 비슷했고, 0.75 이상에서 크게 올랐다. $\rho$=1(무작위)의 성공률은 표 17의 무작위 행({{s:abl:random:success}})과 같다.
  - $\rho$가 작을수록 충돌률은 낮아지고(0.05 수준) 속도 오차는 커져, 5.3절의 위험 편향 붕괴가 연속적으로 나타났다.
  - 최고점은 '대부분 무작위 + 소량의 위험 강화'였다.
- **위험 확률만으로는 이득이 관찰되지 않았다.** $\lambda$=0, $\rho$=0.9(위험 확률 상위 + 저장소)의 성공률({{s:abl:0.0:0.90:success}})은 무작위({{s:abl:random:success}})보다 높지 않았다.
- **불확실성 항 $\lambda$**: 같은 $\rho$에서 $\lambda$=0.5가 $\lambda$=0보다 높았다.
  - $\rho$=0.9: {{s:abl:0.5:0.90:success}} 대 {{s:abl:0.0:0.90:success}}. 계층 부트스트랩 차이의 95% 신뢰구간은 0을 포함하지 않았다({{s:hb:abl_lam05_vs_lam0_r0.90:diff}} [95% CI {{s:hb:abl_lam05_vs_lam0_r0.90:lo}}, {{s:hb:abl_lam05_vs_lam0_r0.90:hi}}], 사후 비교이며 다중 비교 보정은 하지 않음).
  - $\rho$=0.5: {{s:abl:0.5:0.50:success}} 대 {{s:abl:0.0:0.50:success}}
- 반면 불확실성만으로 고른 방법(저장소 없음)은 크게 실패했다({{s:driving:uncertainty:0.02:success}}, 5.2절).

즉 이 절제에서 관찰된 이득은 **위험 확률·불확실성·무작위 저장소가 결합된 경우에만** 나타났다. 위험 강화만으로는 이득이 관찰되지 않았고, 불확실성 단독(저장소 없음)은 실패했다. 엔트로피 항이 왜 저장소와 결합할 때만 도움이 되는지는 본 실험으로 설명할 수 없다. 1차 원고의 '점수기가 확신하지 못하는 경계 상황을 추가해 정책이 결정 경계를 배우도록 돕는다'는 설명은 근거 없는 추측이므로 철회한다. 또한 저장소 혼합을 CARE 점수에만 적용했으므로, 이 결합의 효과가 CNN-GRU 맥락 점수 고유의 것인지, '아무 점수로든 소량 강화 + 무작위 저장소' 구조의 것인지는 본 실험으로 분리되지 않는다. 이 질문을 검정하기 위해 확증 실험에 혼합 통제(4.6절 H3)를 두었으나, CARE의 통제군 대비 우위는 유의하지 않아 분리하지 못했다(5.11절 (3)).

테스트에서도 $\rho$=0.9가 탐색 범위 안에서 가장 높았다. 그러나 $\rho$=0.75와의 차이는 시드 변동 범위 안이며({{s:hb:abl_r0.90_vs_r0.75:diff}} [계층 부트스트랩 95% CI {{s:hb:abl_r0.90_vs_r0.75:lo}}, {{s:hb:abl_r0.90_vs_r0.75:hi}}]), 최적점이 격자 경계에 있었다. 확증 실험에서 탐색한 $\rho$=0.95와 $\rho$=0.9의 차이는 판별되지 않았고(5.11절 (4)), $\rho\in(0.95, 1)$은 평가하지 않았다.

### 5.6 개루프 지표는 폐루프 성능을 대신할 수 있는가

방법 × 예산 {{s:ol_cl_n_conditions:0}}개 조건에서 개루프 지표와 폐루프 성공률의 스피어만 순위상관 $r_s$를 구했다(표 18, 그림 6). $p$는 순열 검정(10,000회, 양측)으로 구했다.

| 범위 | 지표 | $r_s$ | $p$ |
|---|---|---|---|
| 전체 {{s:ol_cl_n_conditions:0}}개 조건 | 전체 행동 MAE | {{s:ol_cl_spearman_mae:2}} | < $10^{-4}$ |
| 전체 {{s:ol_cl_n_conditions:0}}개 조건 | 위험 구간 MAE | {{s:ol_cl_spearman_mae_hazard:2}} | {{s:ol_cl_spearman_mae_hazard_p:3}} |
| 예산 2% 안({{s:ol_cl_within2_n:0}}개 방법) | 전체 행동 MAE | {{s:ol_cl_within2_mae:2}} | {{s:ol_cl_within2_mae_p:4}} |
| 예산 2% 안({{s:ol_cl_within2_n:0}}개 방법) | 위험 구간 MAE | {{s:ol_cl_within2_mae_hazard:2}} | {{s:ol_cl_within2_mae_hazard_p:3}} |

표 18. 개루프 지표와 폐루프 성공률의 스피어만 순위상관. 조건별 값은 시드 평균이다. 전체 조건의 첫 행은 순열 10,000회 가운데 관찰값보다 극단적인 값이 없었다.

- **전체 행동 MAE**는 폐루프 성공률과 강한 음의 상관을 보였다. 전체 조건과 예산 2% 안 모두 그랬다. 본 벤치마크에서는 전체 구간 개루프 오차가 폐루프 순위를 잘 따랐다.
- **위험 구간 MAE**는 전체 조건에서 유의한 상관을 보이지 않았고, 부호는 기대와 반대(양)였다. 이는 '상관이 없다'(효과 크기 0)가 아니라 '유의 수준에 이르지 못했다'는 뜻이다.
- **예산 2% 안**에서는 위험 구간 MAE가 성공률과 유의한 **양(+)의 상관**을 보였다. 위험 구간 오차가 작은 방법일수록 폐루프 성공률이 오히려 낮았다는 뜻이다. 예를 들어 오라클은 위험 구간 MAE가 가장 낮았지만({{s:driving:oracle:0.02:mae_hazard}}), 폐루프 성공률은 최하위권이었다.

이 상관에는 해석상 제약이 있다. 첫째, {{s:ol_cl_n_conditions:0}}개 조건은 독립 표본이 아니다. 같은 방법이 여러 예산에 반복되고, 예산 자체가 두 지표를 함께 움직이므로 전체 조건의 순위상관에는 방법 효과와 예산 효과가 섞여 있다. 둘째, 예산 내 상관은 예산 효과를 제거하지만 조건이 {{s:ol_cl_within2_n:0}}개뿐이고, 방법들이 같은 풀과 같은 테스트 세트를 공유하므로 역시 독립이 아니다. 순열 검정의 $p$는 조건을 교환 가능한 단위로 가정한 참고치다.

이 제약을 감안하면, 결론은 다음과 같이 한정된다. **희소 구간만의 개루프 오차로 선별 효과를 판단하면 오도될 수 있다.** 위험 대응을 개선하려고 데이터를 고르면서 위험 구간 개루프 오차를 기준으로 삼으면, 본 실험에서는 폐루프 성능을 떨어뜨리는 선택을 우대하게 되었다. 이는 개루프 평가의 한계를 지적한 주행 연구[76][81][82]의 결론을 데이터 큐레이션 문제로 확장한 결과다.

{{fig:fig_vla_openloop_vs_closedloop.png|그림 6. 조건별 개루프 행동 MAE(왼쪽: 전체, 오른쪽: 위험 구간)와 폐루프 성공률. 점 하나가 방법×예산 조건(시드 평균)이며, 크기는 예산에 비례한다.}}

### 5.7 언어 조건화의 효과

{{table:vla_language_ablation}}

표 19. 언어 입력 절제(테스트, 시드 3개). 평균 headway는 추종 구간의 평균 시간 간격이며, 지시문의 신중/보통/민첩 스타일별로 표시했다. 전문가 행의 언어 입력 열 '-'는 해당 없음(규칙 제어기)을 뜻한다.

언어 입력의 효과는 측정되지 않았다(표 19).

- **전체 데이터(100%)**: 언어 입력이 있는 정책은 자유주행 속도 오차가 {{s:lang:full:lang:speed_error:2}} m/s로, 언어를 뺀 정책({{s:lang:full:nolang:speed_error:2}} m/s)보다 작았다. 정책이 지시문의 목표 속도를 일부 사용한다는 뜻이다. 그러나 성공률({{s:lang:full:lang:success}} 대 {{s:lang:full:nolang:success}})과 추종 간격의 스타일 구분에는 차이가 없었다.
- **예산 2%**: CARE와 무작위 모두 언어 유무에 따른 차이가 시드 편차 범위 안이었다.
  - 무작위 2%는 언어 없는 정책이 오히려 높았다({{s:lang:random:nolang:success}} 대 {{s:lang:random:lang:success}}).
  - 무작위 조건은 시드 편차가 ±0.12 이상이어서 이 차이를 해석하기 어렵다.
  - 이 절제는 같은 선별법·예산·시드 번호로 정책을 다시 학습한 것이므로 5.2절 주 비교의 반복 실험 역할도 한다. 언어 없는 조건에서는 CARE({{s:lang:ours:nolang:success}})가 무작위({{s:lang:random:nolang:success}})보다 높지 않았다.

언어를 뺀 정책도 스타일별 평균 추종 간격이 신중 > 보통 > 민첩 순으로 갈렸다(전체 데이터 {{s:lang:full:nolang:headway_mean:cautious:2}} / {{s:lang:full:nolang:headway_mean:normal:2}} / {{s:lang:full:nolang:headway_mean:brisk:2}} s). 원인을 추적한 결과, 본 벤치마크에서 에피소드 초기 속도가 목표 속도에 비례해 정해지므로(목표 속도 × U(0.85, 1.0)) **지시문 정보의 상당 부분이 초기 상태로 누출**된다는 것을 확인했다.

따라서 본 벤치마크는 언어 조건화의 효과를 측정하지 못하며, 표 19는 '언어가 성능을 높인다'는 근거로 쓰지 않는다. 본 논문의 결과는 '언어를 입력으로 받는 소형 시각운동 정책'에 대한 것이며, 언어 이해가 핵심인 VLA에 대한 결론으로 일반화하지 않는다. 언어 축의 기여는 정책 성능 향상이 아니라, 수집 데이터에 지시문·물리량 서술·행동을 자동 정렬해 VLA 학습 형식으로 제공하는 것(R3, R4)에 한정한다. 언어 의존성을 엄밀히 측정하려면 같은 초기 상태에서 지시문만 바꾸는 대조 설계가 필요하다(6.3절).

### 5.8 실내 이동로봇(AMR) 도메인

{{table:vla_robot}}

표 20. AMR 도메인 결과(예산 2%, 테스트 72 에피소드, 시드 3개). AMR 성공은 주행 중 충돌(자차 > 0.1 m/s)이 없고 진행이 전문가의 80% 이상인 경우이며, 전문가 기준 거리가 1 m 미만인 에피소드는 진행 조건을 면제했다(수정 기준, 4.5절). 점수기 구조, $\lambda$, $\rho$는 주행 도메인에서 정한 값을 그대로 썼다.

AMR 도메인의 결과는 다음과 같다(표 20). 모든 수치는 수정된 성공 기준(4.5절)으로 저장된 에피소드를 다시 판정한 값이다.

**(1) 위험 편향 붕괴는 그대로 재현되었다.** 성공률은 다음과 같다.

- 감속 트리거: {{s:robot:action_trigger:0.02:success}}
- 맥락 이벤트: {{s:robot:event:0.02:success}}
- 오라클: {{s:robot:oracle:0.02:success}}
- 비교: 무작위 {{s:robot:random:0.02:success}}

이 방법들은 모두 서행으로 진행 기준을 넘지 못했다.

**(2) CARE와 무작위의 차이는 유의하지 않았다.**

- 성공률: CARE {{s:robot:ours:0.02:success}}, 무작위 {{s:robot:random:0.02:success}}
- 계층 부트스트랩: {{s:hb:robot_ours_vs_random_b0.02:diff}} [95% CI {{s:hb:robot_ours_vs_random_b0.02:lo}}, {{s:hb:robot_ours_vs_random_b0.02:hi}}]
- 참고(에피소드 재표집): {{b:robot_ours_vs_random_b0.02}}

1차 원고의 성공 기준에서는 CARE가 무작위보다 낮았고, 에피소드 재표집 신뢰구간도 0을 포함하지 않았다. 그러나 그 기준은 robot_crowded 시나리오에서 성립하지 않는 결함이 있었다(4.5절). 결함을 고친 기준에서는 두 구간 모두 0을 포함하며, CARE가 평균적으로 조금 낮은 경향만 남았다. 즉 AMR에서는 CARE의 이득이 나타나지 않았고, 역전도 확인되지 않았다.

주행에서 고른 $\lambda$, $\rho$를 그대로 옮긴 것이 원인일 수 있어 AMR 검증 시나리오에서 같은 격자로 다시 탐색했다. AMR 검증 재탐색에서도 같은 설정이 선택되었고, 재실행 결과가 원 실행과 비트 단위로 일치했다. 다만 AMR 검증에서도 격자의 최고 설정(성공률 {{s:robot_retune_best_val}})은 무작위({{s:robot_retune_random_val}})보다 높지 않았고, 최고점은 다시 격자 경계($\rho$=0.9)였다. 이 검증 성공률은 재탐색 당시의 기준, 즉 1차 원고의 성공 기준(주행 중 무충돌 ∧ 진행 ≥ 0.8×전문가, 1 m 면제 없음)으로 판정한 값이며, 수정 기준(4.5절)으로 다시 판정하지 않았다. 이 재탐색은 테스트 결과를 본 뒤에 시작했다(표 6).

AMR 점수기의 검증 macro-F1({{s:robot_scorer_val_macro_f1}})은 주행({{s:scorer_val_macro_f1}})보다 낮았다. 위험 판별 AUROC는 검증과 AMR 풀에서 각각 {{s:robot_scorer_val_hazard_auroc}}, {{s:robot_pool_scorer_auroc}}에 해당했다.

**(3) AMR 과제는 이 정책에게 데이터 선택 이전에 어려웠다.**

- 전체 데이터(100%)로 학습한 정책의 성공률({{s:robot:full:1.00:success}})은 전문가({{p:robot_expert_success}})의 절반 수준이었다.
- 같은 정책의 주행 중 충돌률은 {{s:robot:full:1.00:collision_moving}}에 달했다. 수정 기준에서 전문가의 남은 실패는 모두 주행 중 충돌이므로 전문가의 주행 중 충돌률은 1 − {{p:robot_expert_success}}에 해당하며, 정책의 충돌률은 이보다 크게 높았다.

AMR 시나리오는 작업자가 측면에서 경로로 걸어 들어오는 상황이 많다. 이런 상황에서는 화면 가장자리의 작은 사람 박스를 64×64 관측에서 식별해야 하고, 반응 거리도 주행보다 짧다. 하위 정책의 표현력이 부족한 영역에서는 소량의 위험 강화로 얻는 이득보다 정상 표본을 10% 줄이는 손해가 더 클 수 있다. 다만 이 설명은 검증하지 않은 가설이다.

따라서 CARE의 이득 징후는 **하위 정책이 충분한 데이터에서 과제를 대체로 해결할 수 있을 때**(주행: 전체 데이터 성공률 {{s:driving:full:1.00:success}}) 소예산 영역에서만 관찰되었다고 한정하는 것이 정확하다.

### 5.9 실주행 영상 개루프 검증

{{table:vla_comma}}

표 21. comma.ai 실주행 영상 개루프 결과(5-fold 블록 교차검증 × 시드 2개). ±는 10개 실행(fold×시드)의 표준편차다. 정책은 실영상 64×64 두 장과 속도로 미래 1초 가속도를 예측하며, 지시문은 고정 문장이어서 언어 조건 정책이 아니다. 제동 시작 예측 AUROC는 현재 제동 중이 아닌 프레임에서 1초 안의 제동 시작을 예측 청크 평균으로 판별한 값이다. 제동 프레임 회수율은 풀 전체 제동 프레임 가운데 선택된 비율이다. '맥락 이벤트(시뮬 점수기)'는 실라벨 없이 합성 데이터로만 학습한 점수기를 쓴 경우다. 전체 데이터(100%) 행의 엣지 가능 열 '-'는 해당 없음을 뜻한다.

실영상 개루프 결과(표 21)는 시뮬레이터 결과의 **한 부분만** 재현했다. 이 실험은 개루프이므로 정책이 실제로 서행하는 붕괴 자체는 관찰할 수 없다. 또한 이 실험의 정책은 모든 표본에 같은 고정 지시문을 넣으므로 언어 조건 정책이 아니며, 언어에 관한 결론은 이 실험에서 끌어내지 않는다.

**(1) 위험 지향 선택의 개루프 트레이드오프는 실영상에서도 나타났다.**

- 감속 트리거와 오라클은 풀 전체 제동 프레임 가운데 많은 몫을 골랐다(제동 프레임 회수율: 감속 트리거 {{p:comma:action_trigger:0.10:hazard_frame_recall:0}}·{{p:comma:action_trigger:0.20:hazard_frame_recall:0}}, 오라클 {{p:comma:oracle:0.10:hazard_frame_recall:0}}·{{p:comma:oracle:0.20:hazard_frame_recall:0}}, 각각 예산 10%·20%).
- 그 결과 제동 구간 MAE는 가장 낮았다.
  - 감속 트리거: {{s:comma:action_trigger:0.10:mae_braking}}(10%), {{s:comma:action_trigger:0.20:mae_braking}}(20%)
  - 오라클: {{s:comma:oracle:0.10:mae_braking}}(10%), {{s:comma:oracle:0.20:mae_braking}}(20%)
- 그러나 전체 MAE는 다른 방법보다 크게 나빴다.
  - 감속 트리거: {{s:comma:action_trigger:0.10:mae}}(10%)
  - 오라클: {{s:comma:oracle:0.10:mae}}(10%)
  - 비교: 무작위 {{s:comma:random:0.10:mae}}
- 위험 구간 위주로 학습한 정책이 정상 주행 구간에서도 감속을 예측하는 경향으로, 시뮬레이터의 위험 편향 붕괴에 대응하는 개루프 트레이드오프다.

**(2) CARE와 무작위의 차이는 이 데이터에서 측정되지 않았다.**

- 전체 MAE: {{s:comma:ours:0.10:mae}} 대 {{s:comma:random:0.10:mae}}(10%), {{s:comma:ours:0.20:mae}} 대 {{s:comma:random:0.20:mae}}(20%)
- 같은 fold·시드끼리 짝지은 차이(10쌍)는 전체 MAE 기준 {{s:cp:ours_vs_random:0.10:mae:diff}} [95% CI {{s:cp:ours_vs_random:0.10:mae:lo}}, {{s:cp:ours_vs_random:0.10:mae:hi}}](10%), {{s:cp:ours_vs_random:0.20:mae:diff}} [{{s:cp:ours_vs_random:0.20:mae:lo}}, {{s:cp:ours_vs_random:0.20:mae:hi}}](20%)로, 두 예산 모두 0을 포함한다.

**(3) 제동 시작 예측은 모든 조건에서 우연 수준(AUROC 0.5 부근)이었다.**

- 예산 10%의 AUROC(무작위 {{s:comma:random:0.10:brake_onset_auroc:2}}, CARE {{s:comma:ours:0.10:brake_onset_auroc:2}}, 감속 트리거 {{s:comma:action_trigger:0.10:brake_onset_auroc:2}}, 오라클 {{s:comma:oracle:0.10:brake_onset_auroc:2}})를 비롯해 모든 조건이 비슷한 범위에 있었다(표 21).
- 전체 데이터(100%)로 학습한 정책의 AUROC도 {{s:comma:full:1.00:brake_onset_auroc}}에 그쳤다.
- 17분(학습 풀 약 6천 프레임)의 실영상과 64×64 관측으로는 사전학습 없는 소형 정책이 제동 시작을 예측하도록 학습되지 않았다.
- 따라서 이 실험은 선별법 간 하위 성능 차이를 가릴 만큼의 분해능이 없다. 실영상에서의 CARE 효과 검증은 더 큰 데이터와 사전학습 시각 인코더가 필요한 과제로 남긴다.

**(4) 합성 데이터 점수기는 회수율만 높았다.**

- 합성 데이터로만 학습한 점수기('시뮬 점수기')는 예산 10%에서 제동 프레임 회수율({{s:comma:event_sim:0.10:hazard_frame_recall}})이 실라벨 초기 라벨 블록(fold당 영상의 1/5)으로 학습한 점수기({{s:comma:event:0.10:hazard_frame_recall}})보다 높았다.
- 그러나 하위 지표에는 뚜렷한 차이가 없었다. 전체 MAE({{s:comma:event_sim:0.10:mae}} 대 {{s:comma:event:0.10:mae}})와 제동 시작 AUROC({{s:comma:event_sim:0.10:brake_onset_auroc}} 대 {{s:comma:event:0.10:brake_onset_auroc}})는 실행 간 표준편차 범위 안이었다. 20%에서는 회수율 차이도 작았다({{s:comma:event_sim:0.20:hazard_frame_recall}} 대 {{s:comma:event:0.20:hazard_frame_recall}}).
- 회수율이 데이터 가치를 보장하지 않는다는 본 논문의 결과(5.2~5.6절)에 비추면, 이 결과는 합성 데이터로 학습한 점수기가 실영상에서도 제동 구간을 어느 정도 가려낸다는 것까지만 보여 준다. 실차 수집 장치에 바로 배치할 수 있다는 근거로는 부족하다.

### 5.10 계산 비용

표 22는 각 선별법을 수집 시점에 적용하는 데 드는 비용이다.

| 항목 | 값 | 비고 |
|---|---|---|
| 맥락 점수기 파라미터 | {{s:cost_scorer_params:0}} | 멀티채널 CNN-GRU, 창 16 |
| 점수기 1회 추론(창 1개, 배치 1) | {{s:cost_scorer_onnx_ms:3}} ms(ONNX Runtime) / {{s:cost_scorer_torch_ms:2}} ms(PyTorch) | x86 CPU 1스레드 |
| 검출기 YOLOv8n | {{s:cost_yolo640_ms:1}} ms(640) / {{s:cost_yolo320_ms:1}} ms(320) | x86 CPU 4스레드. 수집 장치는 검출기를 인지용으로 이미 실행한다고 가정한다 |
| CARE 선택(54만 프레임 풀, 예산 2%) | {{s:cost_ours_select_s:2}} s | 오프라인 전역 상위 $K$ 선택. 스트리밍 선택은 구현하지 않았다 |
| Coreset 선택(같은 풀) | {{s:cost_coreset_s:1}} s | 풀 전체 특징이 필요해 수집 시점에 적용할 수 없다 |
| 오프라인 손실 기준선 | 정책 학습 {{s:cost_offline_train_min:1}}분 + 풀 전체 추론 | 전체 데이터 저장과 정책 학습이 선행되어야 한다 |

표 22. 선별 비용(x86 클라우드 CPU 측정, 4.4절).

맥락 점수기 비용은 YOLOv8n(640 입력)의 약 {{p:cost_ratio_onnx_yolo640:1}}이다(x86 CPU, ONNX Runtime 기준). PyTorch 추론과 320 입력 검출기를 비교하면 약 {{p:cost_ratio_torch_yolo320:1}}이다. 어느 기준에서도 검출기보다 한 자릿수 이상 작지만, 이 수치는 x86 CPU에서 잰 값이다. 배포 대상 임베디드 장치에서의 지연·전력 측정과 스트리밍 선택의 구현은 남은 과제다(6.3절).

### 5.11 확증 실험 결과

확증 실험은 실행 전에 설계·가설·분석 계획을 고정했다(4.6절, `docs/32_확증실험_사전등록.md`, 커밋 `fcd7f3f`). 조건은 다음과 같다.

- 학습 시드: 3~9의 7개(본 실험과 겹치지 않음)
- 평가 세트: 새로 뽑은 147 에피소드(시드 700000번대)
- CARE 설정: 본 실험과 같은 $\lambda$=0.5, $\rho$=0.9(다시 튜닝하지 않음)

조건별 결과는 표 23, 가설 검정은 표 24에 정리했다.

{{table:vla_confirm}}

표 23. 확증 실험 조건별 결과(새 평가 세트, 평균 ± 시드 표준편차, 시드 7개). '선택 데이터 내 위험 비중'은 선택된 프레임 가운데 GT 위험 프레임의 비율이다. 혼합 통제군('저장소 ρ=0.9 + … 점수')은 저장소 비율과 혼합 규칙만 CARE와 같고, 저장소 클립은 방법마다 다르다(4.6절).

{{table:vla_confirm_tests}}

표 24. 사전 등록 가설 검정. 차이는 성공률 차이이고, 신뢰구간은 시드·에피소드 2단계 계층 부트스트랩(10,000회)이다. 단측 p는 차이 ≤ 0인 재표집의 비율이다(H1은 {{s:ct:H1_ours_vs_random_b0.02:p_le0:4}}, 10,000회 중 1회). Holm 보정은 H2·H3a~c에만 적용했으며, H1과 '(탐색)' 행의 보정 p 열은 해당 없음이다. '검출 가능 차이(근사)'는 시드 대응 차이의 표준오차로 근사한 최소 검출 효과(양측 α=0.05, 검정력 0.8, 2.8×표준오차)다. '(탐색)' 행은 사전 등록에서 검정 대상이 아니라고 정한 기술 통계다.

**(1) 1차 가설 H1은 지지되었다(표 24).**

- 예산 2%에서 CARE의 성공률({{s:confirm:ours_b0.02:success}})은 무작위({{s:confirm:random_b0.02:success}})보다 {{s:ct:H1_ours_vs_random_b0.02:diff}} 높았다.
- 계층 부트스트랩 95% CI([{{s:ct:H1_ours_vs_random_b0.02:lo}}, {{s:ct:H1_ours_vs_random_b0.02:hi}}])와 시드 대응 t의 95% CI([{{s:ctt:H1_ours_vs_random_b0.02:lo}}, {{s:ctt:H1_ours_vs_random_b0.02:hi}}])는 모두 0을 포함하지 않았다.
- 탐색 실험(시드 3개)에서 시드 변동 때문에 결론을 내리지 못했던 2% 우위가, 새 시드·새 평가 세트에서 재현되었다.
- 표 23에서 CARE의 충돌률은 무작위의 약 절반이었고, 위험 시나리오 성공률은 높았으며, 속도 오차는 비슷했다.

**(2) 2차 가설 H2(예산 1%)는 지지되지 않았다.** 차이({{s:ct:H2_ours_vs_random_b0.01:diff}})의 방향은 같았지만, 보정 전 단측 $p$({{s:ct:H2_ours_vs_random_b0.01:p_le0:3}})부터 0.05를 넘었고 신뢰구간도 0을 포함했다([{{s:ct:H2_ours_vs_random_b0.01:lo}}, {{s:ct:H2_ours_vs_random_b0.01:hi}}], Holm 보정 $p$={{s:ct:H2_ours_vs_random_b0.01:p_holm:3}}).

**(3) 맥락 점수기 고유의 기여(H3a~c)는 확증되지 않았다.**

- 저장소 비율($\rho$=0.9)과 혼합 규칙은 같게 두고 점수 몫만 다른 신호로 채운 통제군과 비교했다. 저장소 클립 자체는 방법마다 다르다(4.6절). CARE는 세 통제군 모두보다 점추정이 높았다.
  - 감속 트리거 점수: {{s:ct:H3a_ours_vs_mix_trigger:diff}}
  - 오라클 점수: {{s:ct:H3b_ours_vs_mix_oracle:diff}}
  - 불확실성만: {{s:ct:H3c_ours_vs_mix_uncert:diff}}
- 그러나 어느 비교도 유의하지 않았다(Holm 보정 $p$ ≥ {{s:ct:H3a_ours_vs_mix_trigger:p_holm:2}}, 세 신뢰구간 모두 0을 포함). 보정 전 단측 $p$가 0.05 아래였던 것은 감속 트리거 혼합 대비 비교(H3a, {{s:ct:H3a_ours_vs_mix_trigger:p_le0:3}})뿐이었다.
- 관찰된 차이는 각 비교의 검출 가능 차이(근사, 표 24: H3a {{s:ct:H3a_ours_vs_mix_trigger:mde_seed:3}}, H3b {{s:ct:H3b_ours_vs_mix_oracle:mde_seed:3}}, H3c {{s:ct:H3c_ours_vs_mix_uncert:mde_seed:3}})보다 작았다. 따라서 이 결과는 차이가 없다는 증거가 아니라, 이 표본 크기로 차이를 판별하지 못했다는 뜻이다.
- 탐색적으로, 감속 트리거 점수를 저장소와 섞은 통제군도 무작위보다 {{s:ct:X_mix_trigger_vs_random:diff}} 높았으나 신뢰구간은 0을 포함했다([{{s:ct:X_mix_trigger_vs_random:lo}}, {{s:ct:X_mix_trigger_vs_random:hi}}], 사전 등록 외 비교). 같은 방식의 탐색 비교에서 오라클 점수 혼합은 무작위보다 {{s:ct:X_mix_oracle_vs_random:diff}} 높았고(CI [{{s:ct:X_mix_oracle_vs_random:lo}}, {{s:ct:X_mix_oracle_vs_random:hi}}]), 불확실성 혼합은 {{s:ct:X_mix_uncert_vs_random:diff}}(CI [{{s:ct:X_mix_uncert_vs_random:lo}}, {{s:ct:X_mix_uncert_vs_random:hi}}])였다(표 24, 사전 등록 외). 위험 지향 점수(CARE·오라클·감속 트리거) 혼합이 모두 무작위보다 높은 점추정을 보인 것은, 이득이 특정 점수기보다 '큰 저장소 위의 소량 위험 강화'와 관련 있을 가능성을 시사한다. 다만 이는 탐색 비교이며 판별을 위해서는 더 많은 시드가 필요하다.
- 따라서 확증 실험은 CARE의 무작위 대비 우위(H1)를 보였지만, 그 이득이 CNN-GRU 점수기에서 오는지, '정상 분포를 보존하는 큰 무작위 저장소 + 소량의 위험 지향 강화'라는 구조에서 오는지는 판별하지 못했다. 점추정으로는 세 통제군도 무작위보다 높아(표 23) 구조가 이득의 일부를 설명할 가능성이 있으나, 이는 검정되지 않은 관찰이다.

**(4) 저장소 비율의 경계 확인(탐색).** $\rho$=0.95와 $\rho$=0.9의 차이는 신뢰구간이 0을 포함했다({{s:ct:X_rho095_vs_ours:diff}} [95% CI {{s:ct:X_rho095_vs_ours:lo}}, {{s:ct:X_rho095_vs_ours:hi}}]). 이 구간은 약 0.1 크기의 차이를 배제하지 못하므로, 0.9~0.95 구간의 성능이 같다고 결론 내릴 수는 없다. 기술 통계로는 $\rho$=0.95의 성공률({{s:confirm:ours_b0.02_r0.95:success}})도 무작위({{s:confirm:random_b0.02:success}})보다 높았다.

새 평가 세트에서 예산 2% 조건의 성공률은 본 실험 테스트 세트보다 다소 낮았고(무작위 {{s:confirm:random_b0.02:success}} 대 {{s:driving:random:0.02:success}}, CARE {{s:confirm:ours_b0.02:success}} 대 {{s:driving:ours:0.02:success}}), 예산 1% 조건은 오히려 다소 높았다(무작위 {{s:confirm:random_b0.01:success}} 대 {{s:driving:random:0.01:success}}, CARE {{s:confirm:ours_b0.01:success}} 대 {{s:driving:ours:0.01:success}}). 학습 시드와 평가 세트가 함께 바뀌었으므로 차이의 원인은 구분할 수 없으며, 조건 간 비교는 같은 세트 안에서만 해석한다.

**행동 예측 보조 head.** 실주행 영상에서 맥락 모델에 0.5초·1.0초 뒤 가감속 회귀 head를 붙이면, 맥락 macro-F1은 {{noact_ctx_f1}}(행동 head 없음)에서 {{act_ctx_f1}}로 유지되었고, 행동 MAE는 {{act_mae05}}(0.5초)·{{act_mae10}}(1.0초)로 학습셋 평균 예측({{act_mae05_mean}})보다 낮았다. 예측과 정답의 상관계수는 {{act_corr05}}로, 전방 영상의 박스 수준 특징만으로는 자차 행동을 부분적으로만 설명할 수 있다. 따라서 이 head는 정밀한 제어 정책이 아니라 VLA 학습 데이터의 행동 사전(prior)·이상 탐지 신호로 쓰는 것이 적절하다.

{{table:vla_action}}

표 8. 실주행 영상 행동 보조 head(10-fold).

**자동 큐레이션.** 2초 클립 단위로 저장 예산을 정하면, 맥락 모델 점수로 고른 클립은 예산 10/20/30%에서 제동 프레임의 {{cur10_model}}/{{cur20_model}}/{{cur30_model}}를 포함했다(무작위 선택 {{cur10_random}}/{{cur20_random}}/{{cur30_random}}). 제동 이벤트 단위 포함률은 20% 예산에서 {{cur20_model_ev}}(무작위 {{cur20_random_ev}})였다. 같은 저장량으로 학습 가치가 높은 구간을 두 배 이상 모을 수 있다는 뜻이며, 온디바이스 데이터 수집 비용 절감의 근거가 된다.

{{fig:fig_curation_comma.png|그림 4. 실주행 영상에서 저장 예산 대비 제동 프레임 회수율(10-fold 평균).}}

**실내 이동로봇(AMR) 도메인.** 같은 파이프라인을 로봇 시나리오에 적용한 결과는 표 9와 같다.

{{table:robot}}

표 9. 로봇 도메인 테스트 결과(시드 3개). 이벤트는 감속(slow_down)·안전 정지(safety_stop) 계열이다.

로봇 도메인에서도 특징 v2({{robot:mc_cnn_gru_semantic_v2:macro_f1}})가 v1({{robot:mc_cnn_gru_balanced_v1:macro_f1}})보다 높았다. 그러나 재개(resume) 맥락은 모든 모델에서 거의 인식되지 않았고 안전 정지 F1도 낮아, 저속·근거리 상황의 미세한 변화를 박스 특징으로 포착하는 데 한계가 있다. 주행 모델을 그대로 쓴 zero-shot은 macro-F1 {{robot:driving_pretrained_zero_shot:macro_f1}}로 실패했다(속도·거리 스케일과 맥락 정의가 다르기 때문). 반면 로봇 데이터 10%만 쓸 때는 주행 사전학습 후 미세조정({{robot:robot_10pct_finetune_from_driving:macro_f1}})이 처음부터 학습({{robot:robot_10pct_scratch:macro_f1}})보다 높고 시드 편차도 작았다. 경량 맥락 모델이 도메인 간에 "초기화"로서는 전이된다는 근거다.

{{fig:fig_robot_transfer.png|그림 5. 주행 → 실내 이동로봇 전이(로봇 테스트 macro-F1).}}

**VLA 학습용 내보내기.** 위 과정으로 주행 합성 {{export_synthetic_driving_eps}}개 에피소드({{export_synthetic_driving_frames}} 프레임), 로봇 합성 {{export_synthetic_robot_eps}}개({{export_synthetic_robot_frames}} 프레임), 실주행 큐레이션 클립 {{export_comma_curated_eps}}개({{export_comma_curated_frames}} 프레임)를 LeRobot v2 구조로 내보냈다. 각 프레임은 관측(특징·박스·자차 상태·영상 프레임 번호), 언어(과업 지시문, 한/영 맥락 서술), 행동(미래 가감속), 큐레이션 점수를 가진다. 내보낸 주행 합성 데이터의 실제 서술 예: "{{example_narration_ko}}" / "{{example_narration_en}}".

### 6.1 합성 6맥락 벤치마크

{{fig:fig_synthetic_macro_f1.png|그림 2. 합성 벤치마크 테스트 macro-F1(에피소드 단위 분할, 시드 3개 평균 ± 표준편차). 파랑: 특징 v2, 주황: 특징 v1(2026-03), 회색: 비학습 기준선.}}

{{table:synthetic_main}}

표 2. 합성 벤치마크 테스트 결과(평균 ± 표준편차, 시드 3개). CI는 에피소드 bootstrap 95% 구간(시드별 평균), 지연은 제동 이벤트 시작 대비 첫 검출 시각의 중앙값이다. 정답 라벨 자체의 전환 빈도는 분당 {{gt_flicker}}회다.

**(1) 정확도와 macro-F1은 반대 방향을 가리킨다.** 2026-03 구성(v1 특징, 균등 분할)의 정확도는 {{syn:mc_cnn_gru_balanced_v1:main:accuracy}}로 특징만 v2로 바꾼 같은 구조({{syn:mc_cnn_gru_balanced_v2:main:accuracy}})보다 높지만, macro-F1은 {{syn:mc_cnn_gru_balanced_v1:main:macro_f1}} 대 {{syn:mc_cnn_gru_balanced_v2:main:macro_f1}}로 뒤집힌다. 표 3처럼 v1 구성은 다수 클래스(정상 주행, 추종) F1이 높고 제동 계열 F1이 낮다. 2026-03 보고처럼 정확도를 주 지표로 쓰면 희소 이벤트 성능을 놓친다.

{{table:synthetic_per_class}}

표 3. 클래스별 F1(테스트, 시드 평균).

**(2) 특징 v2가 가장 큰 개선 요인이다.** 같은 멀티채널 구조에서 특징만 v1에서 v2로 바꾸면 macro-F1이 {{syn:mc_cnn_gru_balanced_v1:main:macro_f1}}에서 {{syn:mc_cnn_gru_balanced_v2:main:macro_f1}}로(균등 분할), 제동 이벤트 recall이 {{synm:mc_cnn_gru_balanced_v1:main:event_recall}}에서 {{synm:mc_cnn_gru_balanced_v2:main:event_recall}}로 오른다. 시간 정보가 없는 MLP({{syn:mlp_last_v2:main:macro_f1}})보다 GRU({{syn:gru_v2:main:macro_f1}}), CNN-GRU 계열이 높아 시간 맥락의 효과도 확인된다.

**(3) 의미 기반 채널 그룹의 이점은 확인되지 않았다.** v2에서 의미 그룹({{syn:mc_cnn_gru_semantic_v2:main:macro_f1}})은 균등 분할({{syn:mc_cnn_gru_balanced_v2:main:macro_f1}}), 단일채널({{syn:cnn_gru_single_v2:main:macro_f1}})과 시드 편차 범위 안에서 차이가 없었다. 2026-03 원고의 "의미 단위 채널 분리" 주장은 이 결과로 뒷받침되지 않는다. 대신 창 길이를 16프레임으로 늘린 구성이 {{syn:mc_cnn_gru_semantic_v2_w16:main:macro_f1}}로 가장 높았다.

**(4) 하이브리드 룰 게이트는 macro-F1을 높이지 않았다.** 검증셋에서만 임계값을 고르면, 게이트는 제동 이벤트 recall을 {{synm:mc_cnn_gru_balanced_v1:main:event_recall}}에서 {{synm:mc_cnn_gru_balanced_v1_hybrid:main:event_recall}}로 올리는 대신 오경보를 {{synm:mc_cnn_gru_balanced_v1:main:false_alarms_per_min:2}}에서 {{synm:mc_cnn_gru_balanced_v1_hybrid:main:false_alarms_per_min:2}}회/분으로 늘렸고, macro-F1은 {{syn:mc_cnn_gru_balanced_v1_hybrid:main:macro_f1}}였다. 2026-03에 보고한 brake-critical recall 1.0은 검증셋 튜닝·보고 중복과 소표본(당시 문헌형 모델 recall 0.1667=1/6로 미루어 검증셋 제동 표본 약 6개로 추정)의 영향으로 판단한다. 같은 특징의 룰 단독은 macro-F1 {{synm:rule_v1:main:macro_f1}}에 분당 라벨 전환이 {{synm:rule_v1:main:flicker_per_min:0}}회로, 프레임별 판단의 불안정성이 컸다.

**(5) 손실·후처리.** CB-focal 손실은 제동 계열 F1을 높였지만(표 3) ECE가 {{synm:mc_cnn_gru_semantic_v2_focal:main:ece}}로 커졌다. 인과 EMA 평활화는 분당 라벨 전환을 {{synm:mc_cnn_gru_semantic_v2:main:flicker_per_min:1}}에서 {{synm:mc_cnn_gru_semantic_v2_ema:main:flicker_per_min:1}}회로 줄였으나 여전히 정답(분당 {{gt_flicker}}회)보다 많다. 모든 학습 모델의 ECE는 temperature scaling 후 0.05 이하였다.

### 6.2 강건성: FPS와 검출 노이즈

{{fig:fig_fps_robustness.png|그림 3. 학습 15fps 모델을 10/15/30fps로 재생성한 같은 테스트 상황에 적용했을 때의 macro-F1.}}

{{table:synthetic_robustness}}

표 4. 테스트 조건 변화에 따른 macro-F1.

특징 v2는 단위 테스트에서 FPS와 무관한 역 TTC 값을 냈지만(3.2절), 모델 수준에서는 30fps에서 v2 구성의 하락({{synm:mc_cnn_gru_semantic_v2:main:macro_f1}} → {{synm:mc_cnn_gru_semantic_v2:test_30fps_mid:macro_f1}})이 v1 구성({{synm:mc_cnn_gru_balanced_v1:main:macro_f1}} → {{synm:mc_cnn_gru_balanced_v1:test_30fps_mid:macro_f1}})보다 컸다. 8프레임 창이 담는 시간이 0.53초에서 0.27초로 줄어, 시간 변화를 많이 쓰는 v2 모델이 더 영향을 받은 것으로 본다. 표 5의 보충 실험에서 30fps 입력을 2프레임 간격으로 골라 학습 때와 같은 시간 길이를 보게 하면(시간 기준 창) 이 하락이 줄어드는지 확인하였다. 검출 노이즈를 두 배로 키우면 v2 구성은 {{synm:mc_cnn_gru_semantic_v2:test_15fps_high:macro_f1}}로 떨어져 v1 구성({{synm:mc_cnn_gru_balanced_v1:test_15fps_high:macro_f1}})보다 낮아졌다. 박스 흔들림이 크기 변화율(역 TTC) 추정을 직접 교란하기 때문이며, 추적 필터 강화가 필요하다.

{{section:results_extra_ko}}

### 6.3 실주행 영상 검증

{{table:comma_main}}

표 6. comma.ai speedchallenge 실주행 영상의 자차 운동 상태 추정(10-fold 블록 교차검증 평균 ± 표준편차). 라벨은 속도 센서에서만 계산했다.

실영상에서도 특징 v2의 효과가 뚜렷했다. 같은 구조에서 v1 특징은 macro-F1 {{comma:mc_cnn_gru_balanced_v1:test_20fps:macro_f1}}, 제동 AUROC {{comma:mc_cnn_gru_balanced_v1:test_20fps:braking_auroc}}였고, v2 특징은 {{comma:mc_cnn_gru_balanced_v2:test_20fps:macro_f1}}, {{comma:mc_cnn_gru_balanced_v2:test_20fps:braking_auroc}}였다. 가장 높은 구성은 창 16의 의미 그룹 모델({{comma:mc_cnn_gru_semantic_v2_w16:test_20fps:macro_f1}})이지만 fold 간 표준편차가 0.1 안팎으로 커서 구조 간 차이는 통계적으로 구분되지 않는다. 룰 단독은 제동 이벤트 recall이 높으나({{commam:rule_v2_ego:test_20fps:event_recall}}) 분당 오경보가 {{commam:rule_v2_ego:test_20fps:false_alarms_per_min:1}}회로 실용성이 낮다. 가속 상태는 모든 모델에서 F1이 낮았는데, 전방 장면만으로는 자차의 가속 의도를 관측하기 어렵기 때문이다. 짝수 프레임만으로 특징을 다시 계산한 10fps 평가에서도 macro-F1이 유지되었다(표 6 마지막 열).

{{table:comma_s2r}}

표 7. 합성 데이터로만 학습한 6맥락 모델을 실영상에 그대로 적용했을 때 제동 구간 판별 AUROC(점수 = P(경고)+P(강한 제동)).

합성 데이터로만 학습한 v2 모델은 실영상 제동 구간을 AUROC {{s2r_v2}}로 판별해(v1 모델 {{s2r_v1}}), 물리 기반 합성 데이터가 실제 영상으로 일부 전이됨을 보였다. 다만 역 TTC 특징 하나만으로도 {{s2r_feat}}가 나와, 현재 판별력의 대부분은 학습 모델보다 특징 설계에서 온다.

### 6.4 피지컬 AI / VLA 연계

{{section:results_vla_ko}}

### 6.5 지연시간과 모델 크기

{{table:latency_temporal}}

표 10. temporal 모델 지연시간(창 1개, 배치 1). 측정 장치: {{cpu_model}}({{cpu_count}} 논리 코어). Jetson Orin Nano는 미측정.

{{table:latency_yolo}}

표 11. YOLOv8n(3클래스) CPU 지연시간.

temporal 모델은 모든 구성이 수만 개 파라미터, CPU 단일 스레드에서 1 ms 안팎으로, 파이프라인 지연은 검출기가 지배한다. 클라우드 CPU에서 YOLOv8n(640)은 프레임당 {{lat_yolo_ms}} ms였으며, 실시간 배포에는 GPU/TensorRT 가속이 필요하다.

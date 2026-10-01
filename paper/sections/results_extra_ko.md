{{table:synthetic_extra}}

표 5. 보충 실험(시드 3개): v1+v2 결합 특징과 30fps 시간 기준 창. 같은 설정의 모델을 다시 학습해 측정했으며, 표 2와 같은 구성(균등 분할 v1/v2, 의미 그룹 v2)은 동일한 수치가 재현되었다.

**시간 기준 창 가설은 기각되었다.** 30fps 입력을 2프레임 간격으로 골라 학습 때와 같은 0.53초를 보게 해도 v2 의미 그룹 모델의 macro-F1은 {{synx:mc_cnn_gru_semantic_v2:test_30fps_mid:macro_f1}}(연속 창)에서 {{synx:mc_cnn_gru_semantic_v2:test_30fps_mid_dilated:macro_f1}}(간격 창)로 회복되지 않았다. 따라서 30fps 하락의 주원인은 창 길이가 아니라 특징 계산 단계로 보인다. 프레임마다 독립인 박스 흔들림은 Δt가 짧을수록 변화율(Δln w/Δt) 잡음을 키우는데, 같은 메커니즘이 노이즈 증가 실험의 하락(표 4)과도 일치한다. 변화율을 고정 시간 간격으로 계산하거나 칼만 필터로 추정하는 개선이 필요하다.

**v1과 v2는 상보적이다.** 두 특징을 이어 붙인 32차원 입력은 macro-F1 {{synx:mc_cnn_gru_semantic_v1v2:main:macro_f1}}(창 8), {{synx:mc_cnn_gru_semantic_v1v2_w16:main:macro_f1}}(창 16)로 모든 구성 중 가장 높았고, 정확도({{synxm:mc_cnn_gru_semantic_v1v2_w16:main:accuracy}})도 v1 구성 수준을 유지했다. 고노이즈 조건에서도 {{synxm:mc_cnn_gru_semantic_v1v2_w16:test_15fps_high:macro_f1}}로 v2 단독({{synxm:mc_cnn_gru_semantic_v2:test_15fps_high:macro_f1}})보다 강건했다. v1의 정적 장면 요약(ROI 위험도, 클래스 구성)과 v2의 동적 접근 신호(역 TTC)가 서로 다른 맥락을 담당하기 때문이다.

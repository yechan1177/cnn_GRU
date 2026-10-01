{{table:synthetic_extra}}

표 5. 보충 실험(시드 3개): v1+v2 결합 특징과 30fps 시간 기준 창. 같은 설정의 모델을 다시 학습해 측정했다.

시간 기준 창을 쓰면 30fps에서 v2 의미 그룹 모델의 macro-F1은 {{synx:mc_cnn_gru_semantic_v2:test_30fps_mid:macro_f1}}(연속 창)에서 {{synx:mc_cnn_gru_semantic_v2:test_30fps_mid_dilated:macro_f1}}(2프레임 간격 창)로, v1 구성은 {{synx:mc_cnn_gru_balanced_v1:test_30fps_mid:macro_f1}}에서 {{synx:mc_cnn_gru_balanced_v1:test_30fps_mid_dilated:macro_f1}}로 바뀌었다. v1과 v2를 이어 붙인 32차원 특징은 macro-F1 {{synx:mc_cnn_gru_semantic_v1v2:main:macro_f1}}(창 8), {{synx:mc_cnn_gru_semantic_v1v2_w16:main:macro_f1}}(창 16)였다.

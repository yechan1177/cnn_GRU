| 방법 | 클립 점수 | 필요 정보 | (C1) 인과 | (C2) 정책 비의존 | (C3) 상수 비용 | 엣지 가능 |
|---|---|---|---|---|---|---|
| 무작위 | 균일 난수 | 없음 | O | O | O | O |
| 등간격 | 에피소드별 등간격 | 없음 | O | O | O | O |
| 감속 트리거 | $\max_{t\in c}(-a_t)$ | 자차 가속도(CAN/IMU) | O | O | O | O |
| 역 TTC 규칙 | $\max_{t\in c}\widehat{\mathrm{TTC}}^{-1}_t$ | 검출 + 추적 특징 | O | O | O | O |
| 불확실성 | $\frac{1}{L}\sum_{t\in c}u_t$ | 점수기 | O | O | O | O |
| 맥락 이벤트 | $\max_{t\in c}e_t$ | 점수기 | O | O | O | O |
| **CARE(제안)** | $\max e_t+\lambda\,\overline{u}_t$ + 저장소 $\rho$ | 점수기 | O | O | O | O |
| Coreset[36] | 특징 k-center 탐욕 | 풀 전체 특징 | X | O | X(풀 크기에 비례) | X |
| 오프라인 손실* | $\frac{1}{L}\sum_{t\in c}\ell_\theta(t)$ | 전체 풀로 학습한 정책 | X | X | X | X |
| 오라클* | GT 위험 프레임 비율 | 특권 GT 라벨 | 해당 없음 | O | 해당 없음 | X |

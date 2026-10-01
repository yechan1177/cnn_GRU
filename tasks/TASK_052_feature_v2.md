# TASK_052 검출 특징 모듈화 및 FPS 불변 특징 v2

## 체크리스트
- [x] `vcp.features` 분리(Detection/FrameDetections, v1, v2, registry)
- [x] v1이 2026-03 구현과 같은 값을 내는지 회귀 fixture 테스트(최대 오차 1e-6)
- [x] v2: IoU 추적, Δt 정규화 EMA, 역 TTC, VRU 접근 지표
- [x] FPS 불변성 단위 테스트(10/30fps에서 역 TTC ≈ 0.5)
- [x] v1 클래스 비율 값/이름 불일치 문서화
- [x] `spatial.feature_version` 설정

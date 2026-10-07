# TASK_057 (A4) VLA-lite 소형 정책

계약: `docs/28a_VLA_모듈_인터페이스_계약.md`의 "A4 소유" 절. 설계 문서: `docs/31_VLA_lite_정책.md`.

## 체크리스트
- [x] `src/vcp/vla/policy.py`: PolicyConfig, VLALitePolicy(CNN + FiLM 언어 + proprio)
- [x] ImageSource 프로토콜, PoolImageSource(지연 렌더링 + LRU 상한), ArrayImageSource
- [x] PolicyData: t−2 그룹 경계 규칙, 그룹 내 행동 청크 타깃
- [x] train_policy(고정 steps, AdamW, cosine LR, 시드 고정, 증강)
- [x] make_policy_fn, predict_open_loop
- [x] `tests/test_vla_policy.py`
- [x] 파라미터 수·CPU 1/4스레드 처리량 측정
- [x] `docs/31_VLA_lite_정책.md`
- [x] A2 모듈(obs/render/instructions) 실제 연결 확인

## 결과 메모
- 구조 축소 결정: 3×3 s2 4단(32-64-96-128) → 4×4 s4 패치 줄기 + 3×3 s2 2단(32-64-128). 1스레드 처리량 약 300 → 약 1,150 samples/s(공유 CPU 측정).
- 파라미터: 517,736(어휘 64) / 519,336(A2 VOCAB 89) / 484,520(언어 절제).
- 1스레드 3000×128 실측 336초(소형 풀, 지연 렌더링 포함).
- A2 obs/render/instructions/pool/closed_loop와 연결 확인 완료(소형 풀 → 학습 → 개루프 → 폐루프 4 에피소드).

# TASK_057 (A4) VLA-lite 소형 정책

계약: `docs/28a_VLA_모듈_인터페이스_계약.md`의 "A4 소유" 절. 설계 문서: `docs/31_VLA_lite_정책.md`.

## 체크리스트
- [ ] `src/vcp/vla/policy.py`: PolicyConfig, VLALitePolicy(CNN + FiLM 언어 + proprio)
- [ ] ImageSource 프로토콜, PoolImageSource(지연 렌더링 + LRU 상한), ArrayImageSource
- [ ] PolicyData: t−2 그룹 경계 규칙, 그룹 내 행동 청크 타깃
- [ ] train_policy(고정 steps, AdamW, cosine LR, 시드 고정, 증강)
- [ ] make_policy_fn, predict_open_loop
- [ ] `tests/test_vla_policy.py`
- [ ] 파라미터 수·CPU 1/4스레드 처리량 측정
- [ ] `docs/31_VLA_lite_정책.md`
- [ ] A2 모듈(obs/render/instructions) 실제 연결 확인

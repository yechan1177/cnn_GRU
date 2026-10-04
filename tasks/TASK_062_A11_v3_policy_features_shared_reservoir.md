# TASK_062 (A11) v3 정책 특징 토큰(P1)·공유 저장소 선별(M1)

계획: `docs/33_v3_핵심지표_벤치마크_개선계획.md`. 계약: `docs/28a_VLA_모듈_인터페이스_계약.md`의 "v3 추가 계약"(공통·A11 절).
소유 범위: `src/vcp/vla/policy.py`, `src/vcp/vla/curation.py`(A10 소유 `src/vcp/sim/*`, `closed_loop.py`, `pool.py`는 수정하지 않는다).

## 체크리스트
- [ ] policy.py: `PolicyConfig.use_features/feature_dim/feature_hidden/feature_noise`
- [ ] policy.py: `VLALitePolicy` 특징 MLP(2·D → H → H, clamp(−3,3)) + 헤드 결합, 기본값 False에서 기존과 동일
- [ ] policy.py: `PolicyData.features`, `observation`의 `"features"` [B,2,D](t, t−2 규칙)
- [ ] policy.py: `train_policy`·`make_policy_fn`·`predict_open_loop` 특징 전달·누락 오류
- [ ] curation.py: `select_shared`(공유 저장소 + 점수 몫), 기존 `select`/`METHODS` 유지
- [ ] `tests/test_vla_policy_v3.py`
- [ ] `tests/test_curation_shared.py`
- [ ] 파라미터 수·1스레드 처리량 측정(특징 on/off, 배치 128, 200단계)
- [ ] 문서: `docs/31_VLA_lite_정책.md` v3 절, `docs/30_큐레이션_방법_정의.md` 공유 저장소 절
- [ ] 전체 테스트 통과

## 결과 메모

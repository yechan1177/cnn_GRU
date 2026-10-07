# TASK_062 (A11) v3 정책 특징 토큰(P1)·공유 저장소 선별(M1)

계획: `docs/33_v3_핵심지표_벤치마크_개선계획.md`. 계약: `docs/28a_VLA_모듈_인터페이스_계약.md`의 "v3 추가 계약"(공통·A11 절).
소유 범위: `src/vcp/vla/policy.py`, `src/vcp/vla/curation.py`(A10 소유 `src/vcp/sim/*`, `closed_loop.py`, `pool.py`는 수정하지 않는다).

## 체크리스트
- [x] policy.py: `PolicyConfig.use_features/feature_dim/feature_hidden/feature_noise`
- [x] policy.py: `VLALitePolicy` 특징 MLP(2·D → H → H, clamp(−3,3)) + 헤드 결합, 기본값 False에서 기존과 동일
- [x] policy.py: `PolicyData.features`, `observation`의 `"features"` [B,2,D](t, t−2 규칙)
- [x] policy.py: `train_policy`·`make_policy_fn`·`predict_open_loop` 특징 전달·누락 오류
- [x] curation.py: `select_shared`(공유 저장소 + 점수 몫), 기존 `select`/`METHODS` 유지
- [x] `tests/test_vla_policy_v3.py`(7개)
- [x] `tests/test_curation_shared.py`(19개)
- [x] 파라미터 수·1스레드 처리량 측정(특징 on/off, 배치 128, 200단계)
- [x] 문서: `docs/31_VLA_lite_정책.md` 10절, `docs/30_큐레이션_방법_정의.md` 6절, 28a A11 구현 메모
- [x] 전체 테스트 통과(110개)

## 결과 메모
- 파라미터(어휘 89): 특징 끔 519,336(v2와 같음), 특징 켬 544,040(+24,704).
- `use_features=False`는 v2와 초기 가중치·학습 결과가 비트 단위로 같다(테스트로 확인).
- 1스레드 처리량(배치 128, 200단계, 2회): 증강 켬 기준 특징 끔 1,655/1,695, 특징 켬 1,671/1,716 samples/s. 차이는 반복 편차 안.
- 합성 문제(영상 상수, 특징 한 차원 → 가속도): 특징 모델 MAE/기준선 약 0.11, 특징 없는 모델 약 1.0.
- `select_shared(None, ...)`는 같은 rng 상태의 `select("random", ...)`와 같은 클립 집합이다(ρ 무관).

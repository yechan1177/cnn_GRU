# TASK_061 (A10) v3 시뮬레이터 벤치마크 수정(B1·B2·B3)·폐루프 특징 관측

계획: `docs/33_v3_핵심지표_벤치마크_개선계획.md`. 계약: `docs/28a_VLA_모듈_인터페이스_계약.md`의 "v3 추가 계약"(공통·A10 절).
소유 범위: `src/vcp/sim/env.py`, `src/vcp/vla/pool.py`, `src/vcp/vla/closed_loop.py`, `src/vcp/vla/obs.py`
(A11 소유 `policy.py`, `curation.py`는 수정하지 않는다).

## 체크리스트
- [x] env.py: `collision_pushback`(B3), `decouple_initial_speed`(B1) 키워드 인자, 기본값에서 fixture 비트 동일
- [x] obs.py: `OnlineFeatureTracker`(v1+v2 32차원 온라인 특징, pool·폐루프 공용)
- [x] pool.py: `PoolConfig` 플래그 2개 → SimEnv 전달·meta config 기록, 특징 계산을 공용 헬퍼로 교체(출력 불변)
- [x] closed_loop.py: `make_test_specs(..., counterfactual)`(B2), `run_closed_loop` 플래그 전달, obs `"features"` [B,2,32], 결과 `"counterfactual"`
- [x] `tests/test_vla_env_v3.py`
- [x] 측정: 로봇 전문가 72개(밀어내기 False/True), 주행 전문가 147개(False/True), 특징 계산 추가 후 폐루프 속도
- [x] 문서: `docs/29_VLA_시뮬레이터_폐루프.md` v3 절
- [x] 전체 테스트 통과

## 결과 메모
- 기본 플래그에서 fixture 비트 동일, HEAD `pool.py`·`closed_loop.py`와 출력 동일(검증 스크립트: `experiments/exp_200_vla_sim/v3/measure_v3.py`).
- 로봇 전문가 72개: pushback=False에서 이동 거리 음수 9 → 0, 충돌률 0.181 → 0.181, 주행 중 충돌률 0.056 → 0.097(robot_crowded 측면 진입 판정, 문서 29의 9.6절).
- 주행 전문가 147개: 충돌 0이라 pushback 영향 없음(이동 거리 147개 모두 동일).
- 특징 obs 추가 후 가짜 정책 147개: 벽시계 14.2~15.1 s → 16.6~17.0 s(특징 갱신 약 2.5 s).
- 상세: `docs/29_VLA_시뮬레이터_폐루프.md` 9절, `experiments/exp_200_vla_sim/v3/a10_measurements.json`.

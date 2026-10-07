# TASK_059 A2 단계형 시뮬레이터·렌더러·지시문·풀·폐루프

- 상위 태스크: `tasks/TASK_060_vla_curation_paper.md`
- 계약: `docs/28a_VLA_모듈_인터페이스_계약.md`(A2 소유 부분)
- 설명 문서: `docs/29_VLA_시뮬레이터_폐루프.md`

## 체크리스트
- [x] 계약·기존 코드(`world.py`, `camera.py`, `labels.py`, `dataset.py`, `features/*`) 확인
- [x] 리팩터링 전 회귀 fixture 생성(`tests/fixtures/sim_regression.json.gz`, 13시나리오 × 2시드 × 8초)
- [x] `vla/instructions.py`: 스타일 3종(도메인별), 패러프레이즈 4개(영/한), VOCAB, `encode_instruction`
- [x] `sim/env.py`: `SimEnv` reset/step, 스타일 덮어쓰기, `expert_command`, 외부 제어, `StepInfo`(gt_boxes, horizon_y)
- [x] `world.simulate_episode`를 SimEnv로 재구현, fixture와 비트 동일 확인
- [x] 클래스 id(person=0, vehicle=1, bike=2) 검출 시뮬레이터·특징과 일치 확인(테스트)
- [x] `sim/render.py`: 64×64 렌더러, 속도 측정(프레임당 약 0.09~0.11 ms)
- [x] `vla/obs.py`: 계약 규격 구현
- [x] `vla/pool.py`: 병렬 생성·로드, CSR 박스 저장, 1200 에피소드 시간·크기 외삽(약 34 s, 66 MB)
- [x] `vla/closed_loop.py`: 균형 테스트 사양, lockstep 배치 폐루프, 지표·집계, 전문가 147 에피소드 실행(11.8 s)
- [x] 테스트 `tests/test_vla_env.py`(10건) 및 전체 테스트 통과
- [x] 문서 `docs/29_VLA_시뮬레이터_폐루프.md`, 결과 `experiments/exp_200_vla_sim/`

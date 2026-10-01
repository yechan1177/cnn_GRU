# 29. VLA 시뮬레이터·렌더러·지시문·폐루프 평가(A2)

이 문서는 VLA 연계 논문(문서 28) 검증용 시뮬레이터 측 모듈을 설명한다. 인터페이스는 `docs/28a_VLA_모듈_인터페이스_계약.md`를 따른다.

| 모듈 | 역할 |
|---|---|
| `src/vcp/sim/env.py` | 단계형 환경 `SimEnv`(reset/step), 외부 가속도 명령 제어 |
| `src/vcp/sim/world.py` | 시나리오 정의. `simulate_episode`는 `SimEnv`로 다시 구현(출력 동일) |
| `src/vcp/sim/render.py` | GT 투영 박스 → 64×64 RGB 도식 프레임 |
| `src/vcp/vla/obs.py` | 정책 관측 규격(프레임 2장 스택, proprio, 정규화 상수) |
| `src/vcp/vla/instructions.py` | 스타일 3종, 지시문 패러프레이즈 4개(영/한), VOCAB·토큰화 |
| `src/vcp/vla/pool.py` | 스타일 지시문이 붙은 큐레이션 풀 생성·로드 |
| `src/vcp/vla/closed_loop.py` | 테스트 사양 생성, lockstep 배치 폐루프 평가와 지표 |

## 1. 단계형 환경 `SimEnv`

### 1.1 기존 `simulate_episode`와의 동일성
- 기존 프레임 루프를 그대로 옮겼다. 난수 소비 순서는 다음과 같다.
  1. `random.Random(seed)` 생성
  2. 시나리오 객체 배치(`_build_scenario`)
  3. 자차 IDM 파라미터 t_head, a_max, b_comf, s0, 반응 지연
  4. 초기 속도 배율
  5. 프레임마다 피치 노이즈 → 검출 노이즈
- 리팩터링 전 출력으로 회귀 fixture를 만들었다(`tests/fixtures/sim_regression.json.gz`).
  - 범위: 주행 7종 + 로봇 6종 시나리오 × 시드 2개(3, 1234), 8초
  - 저장 항목: 검출 박스(좌표·클래스·신뢰도), 프레임 시각, 라벨, ego_v, ego_a, ttc, meta
- `tests/test_vla_env.py::test_simulate_episode_matches_pre_refactor_fixture`가 비트 단위 동일성을 검사한다.
- 공개 API(`simulate_episode`, `SCENARIO_TYPES`, `ROBOT_SCENARIO_TYPES`, `DRIVING`, `ROBOT`, `Episode` 등)는 그대로다.

### 1.2 프레임 규약
- `reset()`은 프레임 0을 만든다.
  - 프레임 0의 물리 구간(1/15초)은 내부 전문가로 진행한다.
  - 반응 지연 큐가 0으로 차 있으므로, 이 구간에 실제로 적용되는 명령은 0이다.
- `step(cmd)` 1회는 다음 프레임 1개를 만든다(물리 서브스텝 2개, 30 Hz).
  - 30초 에피소드는 `reset` 1회와 `step` 449회로 프레임 450개가 된다.
- `step(None)`은 기존 내부 전문가다. 서브스텝마다 IDM을 계산하고 반응 지연 큐를 거쳐 액추에이터 1차 지연(τ=0.3초)을 적용한다.
- `step(cmd)`는 외부 명령을 프레임 동안 유지한다.
  - 반응 지연 큐 없이 액추에이터 1차 지연만 적용한다.
  - 명령은 [−ego_brake_max, ego_accel_max]로 자른다. 주행은 [−8, 3], 로봇은 [−2.5, 1.2] m/s²다.
  - 비유한 값(NaN 등)은 경고를 남기고 0으로 바꾼다.
- `expert_command()`는 현재 상태의 지연 없는 IDM 명령이다. [−brake_max, a_max]로 자르며, 스타일이 있으면 스타일 파라미터를 쓴다.
  - `StepInfo.expert_cmd`는 그 프레임 관측 시점의 같은 값이다.

### 1.3 스타일 적용
- `style`을 주면 뽑아 둔 t_head, a_max, b_comf를 스타일 값으로 덮어쓴다. v0는 `target_speed(v0, style, domain)`로 바꾼다.
- 난수는 스타일이 없을 때와 똑같이 소비한다. 그래서 같은 시드면 스타일과 무관하게 주변 객체 시나리오, s0, 반응 지연이 같다.
- 초기 속도는 목표 속도 × U(0.85, 1.0)다.
- `style=None`이면 `summary()`의 meta가 기존과 같다. 스타일이 있으면 meta에 `style`, `v_target`, `v0_scenario`, `a_max`, `b_comf`를 더한다.

### 1.4 `StepInfo`
계약 필드는 다음과 같다.
- `frame`, `gt_boxes`, `horizon_y`
- `ego_v`, `ego_a`, `ego_x`
- `label`, `ttc`, `collided`
- `t`, `gap`, `closing_speed`, `expert_cmd`

계약 외 추가 필드(하위 호환, 기본값 있음)는 다음과 같다.
- `path_is_vehicle`: 경로 객체가 차량인지. 추종 구간 판정에 쓴다.
- `collided_moving`: 자차 속도가 0.5 m/s(로봇은 0.1 m/s)를 넘는 상태에서 충돌했는지.

`gt_boxes`의 규격은 다음과 같다.
- float32 [M,6] = (x1, y1, x2, y2, cls_id, depth_m)
- 원본 640×480 좌표이며, 화면 안으로 자른 GT 투영 박스다.
- 클래스 id는 `camera.KIND_TO_CLS`(person=0, vehicle=1, bike=2)를 따른다. `features.base.CLS_*`, 검출 시뮬레이터와 같다.
  - 노이즈 0 조건에서 검출 박스와 GT 박스의 좌표·클래스가 일치하는지 테스트로 확인한다.

## 2. 렌더러 `render_frame`
- 출력은 uint8 [64,64,3]이다. 원본 640×480을 64×64로 축소하므로 가로세로비가 바뀐다.
- 그리는 순서
  1. 지평선 위는 하늘(주행) 또는 창고 벽(로봇)으로 칠한다.
  2. 지평선 아래는 지면으로 칠한다.
  3. 도로(통로) 사다리꼴을 그린다. 범위는 3차로와 갓길 0.5 m다.
  4. 차선을 그린다. 로봇은 노란 바닥 테이프다.
  5. 객체를 그린다.
- 도로 기하는 핀홀 모델에서 유도한다. 지평선 아래 Δy 행의 횡방향 1 m는 Δy/cam_height 픽셀이다.
  - 사다리꼴과 차선은 `horizon_y`(가감속 피치 + 피치 노이즈)를 따라 움직인다.
- 객체
  - 깊이 내림차순(먼 것부터)으로 칠해 가까운 객체가 덮는다.
  - 색은 person 빨강, vehicle 파랑, bike 주황이다.
  - 깊이에 따라 최대 30%까지 어둡게 한다(최대 거리 기준).
  - 최소 1픽셀로 그려 먼 객체도 남는다.
- 결정성과 속도
  - 난수를 쓰지 않는다.
  - 배경은 지평선을 출력 1/4픽셀 단위로 양자화한 키로 LRU 캐시한다.
  - 프레임당 비용은 배경 복사와 박스 수만큼의 numpy 슬라이싱이다.
- 측정값
  - 조건: 이 저장소의 CPU 컨테이너(4코어), 박스 평균 6.4개, 1350프레임, 3회 측정
  - 결과: 프레임당 **0.09~0.11 ms**(최소 0.092 ms). 목표 0.2 ms 이하다.
- 샘플: `docs/assets/vla_render_samples.png`(사본: `experiments/exp_200_vla_sim/render_samples.png`)
  - 행: 시나리오 8종(free_drive, lead_brake, cut_in, vru_crossing, dense_traffic, robot_agent_stop, robot_human_headon, robot_crowded)
  - 열: 0/6/12/18/24/30초
- 한계
  - 도식 영상이다. 차선이 실선이라 자차 속도의 시각 단서(점선 흐름)는 없다. 속도는 proprio로 준다.
  - 객체 간 깊이 단서는 크기·위치·명암뿐이다.

## 3. 관측 규격(`vla/obs.py`)
- image: [현재 프레임 t, 프레임 max(t−2, 0)]을 채널로 쌓은 uint8 [64,64,6]이다.
- proprio: ego_v / speed_scale이며, speed_scale은 주행 30, 로봇 3이다.
- 행동 정규화: accel_scale은 주행 4, 로봇 1이다.
- 목표 속도는 proprio에 넣지 않고 지시문으로만 준다.

## 4. 지시문(`vla/instructions.py`)

### 4.1 스타일

| 도메인 | 스타일 | t_head(s) | a_max(m/s²) | b_comf(m/s²) | v_factor |
|---|---|---|---|---|---|
| 주행 | cautious | 2.0 | 1.2 | 2.0 | 0.85 |
| 주행 | normal | 1.4 | 1.8 | 2.5 | 1.0 |
| 주행 | brisk | 1.0 | 2.5 | 3.0 | 1.0 |
| 로봇 | cautious | 1.8 | 0.5 | 0.8 | 0.85 |
| 로봇 | normal | 1.2 | 0.75 | 1.0 | 1.0 |
| 로봇 | brisk | 0.8 | 1.0 | 1.2 | 1.0 |

- 로봇 값은 `world.ROBOT`의 무작위 IDM 범위에 맞춰 축소했다. 범위는 t_head 0.8~1.5, a_max 0.5~1.0, b_comf 0.8~1.2다.
- 목표 속도 = 시나리오 v0 × v_factor다.
  - 주행: 10 km/h 단위로 반올림하고 [10, 150] km/h로 제한한다.
  - 로봇: 0.1 m/s 단위로 반올림하고 [0.1, 3.0] m/s로 제한한다.

### 4.2 패러프레이즈와 토큰화
- 패러프레이즈 4개를 영어(모델 입력)와 한국어(meta 기록)로 둔다. 예시는 다음과 같다.
  - 주행 0: "Drive at 60 km/h and keep a short gap to the vehicle ahead, briskly."
  - 주행 1: "Cruise around 50 km/h in a cautious style with a long following distance."
  - 로봇 2: "Target speed 1.2 m/s. Follow the agent ahead with a moderate gap and move smoothly."
- 스타일 표현
  - 간격: long / moderate / short
  - 태도: cautiously / smoothly / briskly
  - 형용사: cautious / normal / sporty
- 토큰화
  - 소문자화 후 단어, 숫자("60", "1.2"), 단위("km/h", "m/s")로 나눈다. 구두점은 버린다.
- `VOCAB`(89개)
  - `<pad>`=0, `<unk>`=1 다음에 아래 토큰을 정렬해 붙인다.
    - 모든 영어 템플릿 단어(두 도메인 × 스타일 3종)
    - 숫자 토큰: 10~150 km/h(10 단위), 0.1~3.0 m/s(0.1 단위)
  - 생성은 결정적이다.
- `encode_instruction(text, max_len=24)`는 int64 [24]를 반환한다. 뒤는 PAD로 채운다. 모든 템플릿 문장에서 UNK가 없음을 테스트로 확인한다.

## 5. 큐레이션 풀(`vla/pool.py`)

### 5.1 생성 규칙
- 시나리오는 `sim.dataset.scenario_for_seed(seed, domain)`로 정한다. 기존 합성 데이터셋과 같은 규칙이다.
- 스타일과 패러프레이즈는 `random.Random(seed·104729 + 3)`로 정한다(`style_for_seed`). 시뮬레이터 난수와 독립이다.
- 에피소드는 `SimEnv(style=...)`를 내부 전문가(`step(None)`, 반응 지연 포함)로 끝까지 돌려 만든다.
- 특징은 `sim.dataset`과 같게 v1/v2로 계산한다(conf 0.45, max_det 30).
- 저장 배열은 계약 28a를 따른다. 추가로 `boundary`(int8, 라벨 전환)를 저장한다.
- GT 박스는 CSR 형식(`box_ptr` [N+1], `boxes` [M,6])으로 저장한다. `frame_boxes(data, idx)`로 꺼낸다.
- meta
  - `episodes`: index, seed, scenario, style, paraphrase, v_target, instruction_en, instruction_ko, frames, 그리고 SimEnv meta
  - 그 밖: `labels`(도메인 라벨 이름), `styles`, `style_names`, `config`, 생성 시간

### 5.2 생성 시간·크기 추정
- 측정 조건: 4코어, workers=4, 30초 에피소드, 압축 npz
- 외삽: 선형

| 풀 | 실측 | 외삽 |
|---|---|---|
| 주행 | 240개, 6.8 s, 13.1 MB | 1200개 약 34 s, 약 66 MB |
| 로봇 | 120개, 3.8 s, 5.9 MB | 600개 약 19 s, 약 30 MB |

- 시드 100000~100095(주행 96개)의 시나리오 분포는 lead_brake 25, cut_in 17, vru_crossing 15, free_drive 13, follow 11, dense 10, stop_and_go 5개였다.
- 스타일 분포는 cautious 36, brisk 31, normal 29개였다.

## 6. 폐루프 평가(`vla/closed_loop.py`)

### 6.1 테스트 사양
- `make_test_specs(domain, n_per_cell, seed_base)`는 시나리오 × 스타일 칸마다 n_per_cell개를 만든다.
  - 시드는 seed_base부터 연속으로 붙인다.
  - 패러프레이즈는 순번 % 4로 돌린다.
  - 시나리오는 시드 규칙이 아니라 사양에서 직접 지정한다.
- 주행은 7×3칸이라 n_per_cell=7이면 147개다. 문서 28의 "140개"를 균형으로 맞추면 147개가 된다.
- 로봇은 6×3칸이라 n_per_cell=4이면 72개다.

### 6.2 실행
- 모든 에피소드를 lockstep으로 진행한다. 프레임 t(0..n−2)마다 아래를 반복한다.
  1. 렌더링
  2. 관측 배치 구성
  3. 정책 1회 호출
  4. `step(cmd)`
- 관측 구성: 현재 프레임과 t−2 프레임을 링 버퍼(슬롯 3개)로 쌓는다.
- `policy_fn=None`(전문가 참조)은 `expert_command()`를 같은 `step(cmd)` 경로로 적용한다.
  - 정책과 액추에이터 경로가 같아 공정하게 비교할 수 있다. 다만 전문가는 정책과 달리 GT 상태를 본다.

### 6.3 지표 정의
에피소드 지표는 다음과 같다(코드 docstring과 같다).
- `collision`: 한 번이라도 경로 객체까지 거리 < 0.3 m가 되었는지
- `collisions_steps`: 충돌한 물리 서브스텝 수
- `collision_moving`: 자차가 움직이는 중(주행 > 0.5 m/s, 로봇 > 0.1 m/s)에 충돌했는지
  - 거의 정지한 자차를 상대가 들이받는 경우를 뺀 자차 기인 충돌의 근사다.
- `min_ttc`: 프레임 경로 TTC의 최솟값이며 10 s로 자른다. TTC가 한 번도 정의되지 않으면 10이다.
- `hazard_success`: 위험 시나리오에서 충돌이 없으면 True다. 그 외 시나리오는 None이다.
  - 주행 위험 시나리오: lead_brake, cut_in, vru_crossing, stop_and_go
  - 로봇 위험 시나리오: robot_agent_stop, robot_human_crossing, robot_human_headon
- `speed_error`: 자유주행 프레임의 |v − v_target| 평균(m/s)
  - 자유주행 프레임: 경로 객체가 없는 프레임
  - 또는 gap > max(40 m(로봇 6 m), 4·T_style·v_target)이면서 TTC가 없거나 8 s를 넘는 프레임
- `headway_error`: 추종 프레임의 |gap/v − T_style| 평균(s)
  - 추종 프레임: 다음을 모두 만족하는 프레임
    - 경로 객체가 차량
    - v ≥ 3 m/s(로봇 0.3 m/s)
    - gap < 2.5·T_style·v
    - |closing| ≤ 1.5 m/s(로봇 0.3 m/s)
  - `headway_mean`(gap/v 평균)도 함께 기록한다.
- `rms_jerk`: 실제 가속도 ego_a의 프레임 차분 jerk RMS(m/s³)
- `distance`: 진행 거리(m)
- `progress`: distance / (v_target·duration)
- `hard_brake_frames`: ego_a < −hard_decel인 프레임 수. hard_decel은 주행 4.0, 로봇 1.2 m/s²다.
- `mean_speed`: 평균 속도(m/s)

집계는 시나리오별·스타일별·전체로 한다.
- 기본은 에피소드 값의 평균이다.
- `hazard_success_rate`는 위험 시나리오 에피소드만으로 계산한다.
- speed/headway 오차는 값이 있는 에피소드만 평균한다.
- `min_ttc`는 평균과 최솟값을 함께 낸다.

### 6.4 IDM headway에 대한 주의
- IDM의 평형 간격은 (s0 + v·T)/√(1 − (v/v0)⁴)다.
- 선행차가 목표 속도에 가까우면 전문가도 T보다 길게 따라간다. 그래서 전문가 `headway_error`는 0이 아니다(주행 약 0.9 s).
- 따라서 정책 평가에서는 전문가 값과의 차이를 본다.
- 스타일 구분은 `headway_mean`으로 확인한다. 아래 6.6절을 본다.

### 6.5 실행 시간
- 측정 조건: 4코어 컨테이너, 단일 프로세스, 30초 에피소드

| 실행 | 에피소드 | 벽시계 시간 | 렌더링 |
|---|---|---|---|
| 주행 전문가 | 147 | 11.8 s | 없음 |
| 로봇 전문가 | 72 | 3.6 s | 없음 |
| 주행 가짜 정책(항상 0 가속) | 147 | 22.1 s | 10.3 s(프레임당 약 0.16 ms, 버퍼 복사 포함) |

- 실제 정책을 쓰면 정책 추론 시간이 더해진다.

### 6.6 전문가 기준 지표
- 시드: 주행 200000+, 로봇 400000+, 노이즈 1.0
- 결과 파일: `experiments/exp_200_vla_sim/`(`.gitignore`의 `experiments/*` 규칙으로 기본 미추적)
  - `expert_closed_loop_driving.json`, `expert_closed_loop_robot.json`
  - `zero_policy_closed_loop_driving.json`, `timing.json`

전체 지표:

| 실행 | 충돌률 | 주행 중 충돌률 | 위험 성공률 | 속도 오차 | headway 오차 | RMS jerk | 진행률 |
|---|---|---|---|---|---|---|---|
| 주행 전문가(147) | 0.000 | 0.000 | 1.000 | 0.48 m/s | 0.90 s | 0.83 | 0.74 |
| 주행 0 가속(147) | 0.592 | 0.592 | 0.298 | 4.20 m/s | 1.09 s | 0.00 | 0.66 |
| 로봇 전문가(72) | 0.153 | 0.042 | 0.917 | 0.03 m/s | 1.44 s | 0.24 | 0.71 |

시나리오별 충돌률(전문가):

| 주행 시나리오 | 충돌률 | 로봇 시나리오 | 충돌률(주행 중) |
|---|---|---|---|
| free_drive | 0.00 | robot_aisle_free | 0.00 (0.00) |
| follow | 0.00 | robot_follow_agent | 0.00 (0.00) |
| lead_brake | 0.00 | robot_agent_stop | 0.00 (0.00) |
| cut_in | 0.00 | robot_human_crossing | 0.00 (0.00) |
| vru_crossing | 0.00 | robot_human_headon | 0.25 (0.08) |
| dense_traffic | 0.00 | robot_crowded | 0.67 (0.17) |
| stop_and_go | 0.00 | | |

스타일별 headway_mean(전문가 추종 프레임 gap/v 평균):

| 도메인 | cautious | normal | brisk |
|---|---|---|---|
| 주행 | 3.14 s | 2.26 s | 1.69 s |
| 로봇 | 3.46 s | 2.66 s | 1.86 s |

- 스타일 지시가 행동 차이로 이어진다.

로봇 충돌에 대한 해석:
- 로봇 전문가의 충돌 대부분은 작업자가 정지한 로봇을 향해 걸어오는 경우다.
  - robot_crowded는 경로 근처 작업자의 무작위 횡이동, robot_human_headon은 늦은 비켜서기가 원인이다.
- 이 충돌은 기존 시뮬레이터 시나리오의 성질이다. 이 문서의 리팩터링이 만든 것이 아니다.
  - 기존 `simulate_episode`에서도 robot_crowded는 충돌 스텝이 있었다.
- 그래서 로봇 평가에서는 `collision_moving`을 함께 본다.
- robot_crowded는 위험 시나리오 성공률에서 뺐다.

## 7. 테스트
`tests/test_vla_env.py`(10건):
- simulate_episode 회귀(fixture 비트 동일)
- SimEnv 외부 제어·수명주기·명령 클립
- 스타일 덮어쓰기와 시나리오 보존
- GT 클래스 id와 검출 일치
- 렌더 형태·결정성·도메인 차이·가림 순서
- obs 보조 함수
- 지시문·VOCAB
- 풀 저장·로드 round-trip
- 테스트 사양 균형
- 폐루프 소규모 실행(0 가속 정책·전문가, 결정성)

## 8. 한계
- 렌더러는 도식 영상이라 실사 영상과 도메인 차이가 크다. 실영상 일반화는 comma 개루프 평가로 따로 본다(A3·문서 28).
- 전문가는 GT 상태를 쓰는 특권 정보 기준이다. 정책 성능의 상한이 아니라 참고 기준이다.
- 외부 제어에서는 반응 지연이 없다.
  - 전문가 참조의 반응은 풀 데모(반응 지연 포함 IDM)보다 빠르다.
  - 정책의 실제 반응 시간은 관측 주기(1/15초)와 액추에이터 지연으로 정해진다.

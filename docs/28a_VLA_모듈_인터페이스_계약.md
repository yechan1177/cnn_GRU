# 28a. VLA 연계 모듈 인터페이스 계약

여러 작업자가 병렬로 구현하므로 아래 시그니처와 배열 규격을 바꾸지 않는다. 바꿔야 하면 이 문서를 먼저 고친다.

## 공통
- 실행 Python: 스크래치 venv(`.../scratchpad/venv/bin/python`). 패키지 `vcp`는 editable 설치되어 있다.
- 테스트: `python -m pytest -q`(저장소 루트).
- 문서·주석·docstring은 한국어, 타입 힌트 필수, 로깅 사용.
- 클래스 id: person=0, vehicle=1, bike=2(기존 YOLO 3클래스와 같다).

## A2 소유
### `src/vcp/vla/obs.py`: 관측 규격(학습·폐루프 공통)
```python
IMG_SIZE: tuple[int, int] = (64, 64)        # (H, W)
HISTORY_OFFSET: int = 2                       # 현재 프레임 t와 t-2 프레임을 채널로 쌓는다
def speed_scale(domain: str) -> float         # driving 30.0, robot 3.0
def accel_scale(domain: str) -> float         # driving 4.0, robot 1.0 (정책 출력 정규화용)
def stack_frames(cur: np.ndarray, prev: np.ndarray) -> np.ndarray   # uint8 [H,W,3]x2 -> uint8 [H,W,6]
def proprio(ego_v: float | np.ndarray, domain: str) -> np.ndarray   # float32 [...,1] = ego_v / speed_scale
```
- 에피소드 초반(t < 2)에는 prev로 프레임 0을 쓴다.
- 목표 속도는 proprio에 넣지 않는다. 지시문(언어)으로만 전달한다.

### `src/vcp/sim/render.py`
```python
def render_frame(boxes: np.ndarray, horizon_y: float, domain: str = "driving",
                 size: tuple[int, int] = IMG_SIZE, cam: CameraConfig | None = None) -> np.ndarray  # uint8 [H,W,3]
```
- `boxes`: float32 [M,6] = (x1, y1, x2, y2, cls_id, depth_m). 원본 카메라 픽셀 좌표(640×480)이며 GT 투영 박스다.
- 하늘·도로(사다리꼴)·차선을 그리고, 객체는 먼 것부터 클래스별 색으로 그린다.
- 결정적(난수 없음)이고 빨라야 한다(목표: 프레임당 0.2 ms 이하).

### `src/vcp/vla/instructions.py`
```python
@dataclass(frozen=True) class DrivingStyle: name: str; t_head: float; a_max: float; b_comf: float; v_factor: float
def styles_for_domain(domain: str) -> dict[str, DrivingStyle]   # "cautious" | "normal" | "brisk"
def sample_style(rng: random.Random, domain: str) -> DrivingStyle
def target_speed(v0: float, style: DrivingStyle, domain: str) -> float   # 주행 10 km/h, 로봇 0.1 m/s 단위로 반올림
N_PARAPHRASES: int = 4
def instruction_text(style_name: str, v_target: float, domain: str, paraphrase: int, lang: str = "en") -> str
VOCAB: dict[str, int]                         # 결정적 생성, 0=PAD, 1=UNK
def encode_instruction(text: str, max_len: int = 24) -> np.ndarray   # int64 [max_len]
```

### `src/vcp/sim/env.py`
```python
class SimEnv:
    def __init__(self, scenario: str, seed: int, fps: float = 15.0, duration_s: float = 30.0,
                 noise: DetectorNoiseConfig | None = None, style: DrivingStyle | None = None,
                 physics_hz: float = 30.0) -> None
    n_frames: int; frame_idx: int; done: bool; v_target: float; instruction_style: str | None
    def reset(self) -> StepInfo
    def expert_command(self) -> float          # 현재 상태의 지연 없는 IDM 명령(스타일 파라미터 사용)
    def step(self, cmd: float | None = None) -> StepInfo
    def summary(self) -> dict[str, Any]
```
- `step(None)`: 기존과 같은 내부 전문가(IDM + 반응 지연 큐)를 물리 서브스텝마다 계산한다.
- `step(cmd)`: 외부 명령을 프레임 동안 유지하고, 액추에이터 1차 지연만 적용한다(반응 지연 큐 없음).
- `StepInfo`(dataclass)
  - `frame: FrameDetections`(노이즈 검출), `gt_boxes: np.ndarray [M,6]`, `horizon_y: float`
  - `ego_v`, `ego_a`, `ego_x`, `label: int`, `ttc: float`(무한대는 -1), `collided: bool`(이번 프레임)
  - `t`, `gap`, `closing_speed`, `expert_cmd: float`(프레임 시점의 지연 없는 IDM 명령)
- `style=None`이면 기존과 똑같은 무작위 IDM 파라미터를 쓴다. `simulate_episode`는 SimEnv로 다시 구현하되, 리팩터링 전과 출력이 비트 단위로 같아야 한다(회귀 fixture).

### `src/vcp/vla/pool.py`
```python
@dataclass class PoolConfig: n_episodes: int; seed_base: int; fps: float = 15.0; duration_s: float = 30.0;
                             noise_level: float = 1.0; domain: str = "driving"; workers: int = 4; conf_threshold: float = 0.45
def generate_pool(out_path: Path, cfg: PoolConfig) -> Path       # <out>.npz + <out>.meta.json
def load_pool(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]
```
- npz 배열(N=전체 프레임)
  - `X_v1`, `X_v2` [N,16] float32(노이즈 검출 특징). 로드할 때 `X_v1v2`를 파생한다.
  - `y` [N] int16, `ep` [N] int32, `t`, `ego_v`, `ego_a`, `ttc`, `expert_cmd`, `horizon_y`, `v_target` [N] float32
  - `box_ptr` [N+1] int64, `boxes` [M,6] float32(GT 투영 박스, 렌더링용)
  - `style_id` [N] int8, `paraphrase` [N] int8
- meta: `episodes`(index, seed, scenario, style, v_target, instruction_en, instruction_ko, collisions_steps …), `labels`, `config`, `styles`.
- 시나리오는 기존 `scenario_for_seed` 규칙을 따른다. 스타일·패러프레이즈는 시드로 결정한다.

### `src/vcp/vla/closed_loop.py`
```python
@dataclass(frozen=True) class EpisodeSpec: scenario: str; seed: int; style: str; paraphrase: int
def make_test_specs(domain: str, n_per_cell: int, seed_base: int) -> list[EpisodeSpec]   # 시나리오 × 스타일 균형
PolicyFn = Callable[[dict[str, np.ndarray]], np.ndarray]
#   입력 배치: {"image": uint8 [B,H,W,6], "tokens": int64 [B,L], "proprio": float32 [B,1]}
#   출력: float32 [B] 물리 단위 가속도 명령(m/s^2)
def run_closed_loop(policy_fn: PolicyFn | None, specs: list[EpisodeSpec], domain: str = "driving",
                    fps: float = 15.0, duration_s: float = 30.0, noise_level: float = 1.0) -> dict[str, Any]
```
- `policy_fn=None`이면 전문가(지연 없는 IDM 스타일 명령)로 같은 시드를 돈다(참고 기준).
- 모든 에피소드를 동시에 진행하며 정책을 배치로 1회 호출한다.
- 에피소드 지표
  - 충돌 여부, 충돌 스텝 수, 최소 TTC
  - 위험 시나리오 성공(충돌 없음)
  - 자유주행 구간 속도 오차 |v − v_target|, 추종 구간 시간 headway 오차 |gap/v − T_style|
  - RMS jerk, 진행 거리
- 집계: 시나리오별·스타일별·전체.

## A3 소유
### `src/vcp/vla/curation.py`(기존 `select_clips`, `event_coverage` 유지)
```python
METHODS: tuple[str, ...] = ("random", "uniform", "action_trigger", "rule_ittc", "uncertainty",
                            "event", "coreset", "offline_loss", "oracle", "ours")
def clip_starts(group: np.ndarray, clip_len: int) -> np.ndarray     # 그룹 경계를 넘지 않는 비중첩 클립 시작점
def select(method: str, budget_ratio: float, group: np.ndarray, clip_len: int,
           rng: np.random.Generator, *, event_score: np.ndarray | None = None,
           entropy: np.ndarray | None = None, features: np.ndarray | None = None,
           action: np.ndarray | None = None, ittc: np.ndarray | None = None,
           oracle: np.ndarray | None = None, loss: np.ndarray | None = None,
           lam: float = 0.5, reservoir: float = 0.3, per_group_cap: int | None = None) -> np.ndarray  # bool [N]
def selection_stats(mask: np.ndarray, labels: np.ndarray, group: np.ndarray,
                    n_classes: int, hazard_ids: tuple[int, ...]) -> dict[str, Any]
```
- `ours`: 점수 `event_score + lam*entropy`, 예산의 (1−reservoir)는 점수 상위, 나머지는 남은 클립에서 무작위로 고른다.
- `action_trigger`는 `-action`(감속 크기)의 클립 최댓값으로 고른다.
- `rule_ittc`는 역 TTC 특징의 클립 최댓값으로 고른다.

### `src/vcp/tools/extract_small_frames.py`
- comma `train.mp4`를 64×64 uint8로 축소해 `frames` [20400,64,64,3]로 저장한다(`data/processed/comma_speedchallenge/frames_64.npz`).
- 하늘·보닛을 자르는 crop 인자를 둔다.
- (A3 추가, 시그니처 변경 없음) npz에 `crop` int32 [4] = (y0, y1, x0, x1), `size` int32 [2] = (H, W), `frame_id` int32 [N], `fps`도 저장한다. 기본 crop은 (100, 360, 0, 640)이며 근거는 모듈 docstring에 있다. `frames[i]`는 `comma_table.npz`의 `frame_id == i` 행과 대응한다(현재 항등 정렬).
- (A3 추가) `curation.ITTC_FEATURE_INDEX` = `V2_KEYS.index("lead_inv_ttc")`. 방법별 정의·계산량은 `docs/30_큐레이션_방법_정의.md`.

## A4 소유
### `src/vcp/vla/policy.py`
```python
@dataclass class PolicyConfig: chunk: int = 8; vis_dim: int = 128; lang_dim: int = 64; use_language: bool = True;
                               steps: int = 3000; batch: int = 128; lr: float = 1e-3; weight_decay: float = 1e-4;
                               seed: int = 0; augment: bool = True; threads: int = 1
class VLALitePolicy(nn.Module)       # CNN(6채널) + 언어 임베딩(FiLM) + proprio → 정규화된 가속도 청크 [B, chunk]
class ImageSource(Protocol):  def get(self, idx: np.ndarray) -> np.ndarray   # uint8 [B,H,W,3]
class PoolImageSource          # pool의 boxes/horizon_y로 지연 렌더링(LRU 캐시)
class ArrayImageSource         # 미리 만든 프레임 배열(comma)
@dataclass class PolicyData: images: ImageSource; group: np.ndarray; ego_v: np.ndarray; action: np.ndarray;
                             tokens: np.ndarray [N,L] int64; domain: str
def train_policy(data: PolicyData, train_idx: np.ndarray, cfg: PolicyConfig) -> tuple[VLALitePolicy, dict]
def make_policy_fn(model: VLALitePolicy, domain: str) -> PolicyFn
def predict_open_loop(model, data: PolicyData, idx: np.ndarray) -> np.ndarray   # [len(idx), chunk] 물리 단위
```
- 학습 대상 = 같은 그룹 안의 `action[t : t+chunk]`(끝은 마지막 값으로 채움), `accel_scale`로 정규화한다.
- 경사 단계 수(`steps`)를 고정해 데이터 양과 계산량을 분리한다.
- 구현 확장(시그니처 변경 없음, 모두 기본값): `PolicyConfig`에 `huber_beta`, `front_weight`, `warmup_steps`, `grad_clip`,
  `n_curve_bins`, `device` 필드, `PoolImageSource(..., size=None, max_cache=100_000)`와 `from_pool(arrays, domain)`,
  `PolicyData.observation(idx)`(PolicyFn 입력 형식 dict), `predict_open_loop(..., batch_size=512)`. 세부는 `docs/31_VLA_lite_정책.md`.

## v3 추가 계약(2026-10-04, docs/33)

### 관측에 검출 특징 토큰 추가(A10·A11 공통)
- `PolicyFn` 입력 dict에 `"features"`: float32 [B, 2, 32]를 추가한다.
  - 프레임 t와 t−HISTORY_OFFSET의 v1+v2 특징을 이어 붙인 것이다(`X_v1v2`와 같은 순서: v1 16 + v2 16).
  - 에피소드 초반 경계 규칙은 영상과 같다(t−2가 없으면 프레임 0을 쓴다).
- 특징은 노이즈 검출(FrameDetections)에서 `build_feature_extractor("v1"/"v2", conf_threshold=0.45, max_det=30)`로 인과적으로 계산한다. pool.py와 같은 방식이다.
- 정책이 특징을 쓰지 않으면(`use_features=False`) 이 키를 무시한다.

### A10(시뮬레이터)
- `SimEnv(..., collision_pushback: bool = True)`: False면 충돌 시 자차를 뒤로 밀지 않는다(속도만 장애물 속도 이하로 제한).
- 스타일 지정 시 초기 속도는 `v0_scenario * U(0.85, 1.0)`이다(목표 속도와 분리). 난수 소비 순서는 그대로다.
  - 하위 호환 플래그: `SimEnv(..., decouple_initial_speed: bool = True)`. 기존 동작은 False.
- `PoolConfig`에 `collision_pushback: bool = True`, `decouple_initial_speed: bool = False`를 추가한다. v3는 False/True로 생성한다.
- `make_test_specs(domain, n_per_cell, seed_base, counterfactual: bool = False)`
  - True면 시드마다 세 스타일 사양을 모두 만든다(시나리오 × 시드 × 3스타일).
- `run_closed_loop(..., collision_pushback: bool = True, decouple_initial_speed: bool = False)`
  - 위 플래그를 환경에 전달하고, obs에 `"features"`를 넣는다.
- 구현 메모(A10, 시그니처 변경 없음, 기본값에서 기존 출력과 동일)
  - 반사실 사양: 시나리오마다 시드를 seed_base부터 연속으로 붙이고, 사양 순서는 시나리오 → 시드 → STYLE_NAMES. 패러프레이즈 = 시드 % 4(세 스타일 공통).
  - 결과 dict: `"counterfactual"`(사양 구조 자동 판정, `closed_loop.is_counterfactual`), `config`에 두 플래그, `timing.features_s`.
  - 특징 공용 헬퍼(`vla/obs.py`): `OnlineFeatureTracker`(에피소드 1개, pool.py도 사용), `FeatureHistory`(폐루프 배치 [B,2,32]),
    `stack_feature_history(feats [T,32], t)`(풀 특징으로 t, t−2 관측 재구성), 상수 `FEATURE_DIM=32`.
  - `SimEnv.summary()`는 플래그가 기본값이 아닐 때만 `collision_pushback`/`decouple_initial_speed`를 meta에 더한다.
  - 상세·측정: `docs/29_VLA_시뮬레이터_폐루프.md` 9절.

### A11(정책·선별)
- `PolicyConfig.use_features: bool = False`, `feature_dim: int = 32`.
  - 특징 [B,2,32]를 평탄화해 MLP(64→64)로 보낸 뒤 헤드 입력에 결합한다.
- `PolicyData.features: np.ndarray | None`([N,32])
  - `observation(idx)`가 t, t−2 규칙으로 `"features"` [B,2,32]를 만든다.
- `curation.select_shared(method_score: np.ndarray | None, budget_ratio, group, clip_len, rng, reservoir: float, entropy=None, lam=0.0) -> mask`
  - 저장소 클립 수 R = round(ρ·K)를 먼저 `rng`로 뽑는다. 시드가 같으면 방법과 무관하게 같은 클립이 뽑힌다.
  - 나머지 K−R개는 남은 클립 가운데 클립 점수(max(score) + λ·mean(entropy)) 상위로 채운다.
  - `method_score=None`이면 무작위로 채운다(무작위 기준선).
- 구현 메모(A11, 시그니처 변경 없음, 기본값으로 계약 동작 유지)
  - `PolicyConfig` 확장 필드: `feature_hidden: int = 64`(특징 MLP 폭), `feature_noise: float = 0.02`(augment=True일 때 특징 가우시안 노이즈 σ).
  - `VLALitePolicy.forward(image, tokens, proprio, features=None)`: features는 use_features일 때만 필수. 입력은 NaN→0, clamp(−3,3) 후 MLP.
  - `make_policy_fn`: use_features 모델에 `"features"`가 없으면 `ValueError`. `predict_open_loop`·`train_policy`도 `PolicyData.features`가 없으면 `ValueError`.
  - `select_shared`: rng 순열 π의 앞 R개가 저장소. None 기준선은 π의 앞 K개(같은 rng 상태의 `select("random")`과 같은 집합). method_score는 1차원만 받는다.
  - 분석 보조: `curation.shared_reservoir_mask(budget_ratio, group, clip_len, rng, reservoir)`(저장소 부분 마스크).
  - 상세: `docs/31_VLA_lite_정책.md` 10절, `docs/30_큐레이션_방법_정의.md` 6절.

## v4 추가 계약(A15)(2026-10-04, docs/36)

공통 계약은 `docs/36_v4_알고리즘_모델_개선계획.md` 6절이다. 아래는 A15 구현 메모다. 모두 기본값에서 v3와 같다.

### 관측 특징 이력(`vla/obs.py`)
- `"features"`: float32 [B, H, 32]. 인덱스 k(0..H−1)는 프레임 max(t − k·s, 0)의 특징이다(에피소드 시작에서 잘라 냄, k=0 현재).
- `FeatureHistory(batch, history=2, stride=2, conf_threshold=0.45, max_det=30)`
  - `update(frames)` → [B, H, 32]. 링 버퍼 슬롯 (H−1)·s+1개.
  - `conf_threshold`·`max_det`는 이제 3·4번째 위치다. 키워드로 넘긴다.
- `stack_feature_history(feats, t, history=2, stride=2)` → [H, D] 또는 [M, H, D]
- `feature_history_offsets(history, stride)`, 상수 `DEFAULT_FEATURE_HISTORY = DEFAULT_FEATURE_STRIDE = 2`
- 영상 관측 (t, t−2) 2장과 `HISTORY_OFFSET`, `history_index`는 바뀌지 않았다.

### 폐루프(`vla/closed_loop.py`)
- `run_closed_loop(..., feature_history: int | None = None, feature_stride: int | None = None)`
  - None이면 `policy_fn.feature_history`/`policy_fn.feature_stride` 속성을 쓰고, 없으면 2/2다(각각 따로 결정).
  - A14 `make_policy_fn`은 반환 함수에 모델 설정의 두 속성을 붙인다.
  - 결과 `config["feature_history"]`, `config["feature_stride"]`에 실제 값을 기록한다.
- `resolve_feature_history(policy_fn, feature_history=None, feature_stride=None) -> (H, s)`(공개 헬퍼)

### 선별(`vla/curation.py`)
- `select_shared(..., per_group_cap: int | None = None, info: dict | None = None)`
  - c가 주어지면 점수 몫(저장소 밖 점수 상위)에서 group(VLA 스위트에서는 에피소드 id `pool["ep"]`)당 최대 c개 클립을 고른다.
  - 상한 때문에 못 채운 자리는 상한을 무시한 다음 점수 순으로 채운다. 선택량은 v3와 같다. `info["n_cap_overflow"]`·INFO 로그로 남긴다.
  - 저장소와 rng 소비는 c와 무관하다. 무작위 기준선(None 점수)은 c를 무시한다.
  - `info` 키: `k`, `n_reservoir`, `n_score`, `per_group_cap`, `n_cap_overflow`, `n_score_groups`

### 학습 쪽과 맞출 점(A14·v4 스위트)
- 학습 `PolicyData.observation(idx, history, stride)`는 group 연속 구간 시작에서 잘라 낸다.
- v3 스위트는 group으로 **클립 id**를 넘긴다. 이대로면 학습 이력은 클립 시작에서, 폐루프 이력은 에피소드 시작에서 잘린다.
  - H=8, s=2에서는 30프레임 클립의 앞 14프레임이 다르다.
  - v4 스위트에서 어느 쪽을 쓸지 명시해야 한다. 상세: `docs/29_VLA_시뮬레이터_폐루프.md` 10.3절.
- 상세·측정: `docs/29_VLA_시뮬레이터_폐루프.md` 10절, `docs/30_큐레이션_방법_정의.md` 7절.

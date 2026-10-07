from __future__ import annotations

"""VLA-lite: 렌더링 영상 + 지시문 + 자차 속도로 미래 가속도 청크를 내는 소형 VLA 정책.

큐레이션 방법의 효과를 "하위 VLA 정책 성능"으로 검증하기 위한 고정 구조·고정 학습 예산 정책이다.
설계·정규화·학습 통제 원칙은 `docs/31_VLA_lite_정책.md`, 인터페이스는
`docs/28a_VLA_모듈_인터페이스_계약.md`(A4 소유 절)를 따른다.

구성 요소
- `VLALitePolicy`: 6채널(t, t−2) CNN + FiLM 언어 조건화 + proprio MLP → 정규화 가속도 청크 [B, chunk]
  (v3: `use_features=True`이면 검출 특징 토큰 [B,2,32] MLP 출력을 헤드 입력에 결합, docs/33 P1)
  (v4, docs/36: 특징 이력 [B,H,D]와 GRU 시간 인코더(P2), 위험 맥락 보조 헤드(P3), 지시문 드롭아웃(T1).
  모두 기본값에서는 v3와 비트 단위로 같다)
- `ImageSource` / `PoolImageSource`(지연 렌더링 + LRU 상한) / `ArrayImageSource`(미리 만든 프레임)
- `PolicyData`: 그룹(에피소드·블록) 경계를 지키는 관측(t−2 규칙, 특징 이력 t−k·s 규칙)·행동 청크 조회
- `train_policy`: 고정 경사 단계(steps) 학습. 데이터 양과 계산량을 분리한다.
- `make_policy_fn`: 폐루프 평가용 배치 정책 함수(계약 `PolicyFn`)
- `predict_open_loop`: 개루프 평가용 청크 예측(물리 단위)

관측 규격(`IMG_SIZE`, `HISTORY_OFFSET`, `accel_scale`, `stack_frames`, `proprio`)은 A2 소유
`vcp.vla.obs`를 지연 import해서 쓴다. 해당 모듈이 아직 없을 때만 계약 값과 같은 대체 구현을
쓰고 경고를 남긴다(병렬 개발 중 독립 검증용).
"""

import logging
import math
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

PolicyFn = Callable[[dict[str, np.ndarray]], np.ndarray]
"""계약의 `vcp.vla.closed_loop.PolicyFn`과 같은 형식(순환 import를 피하려고 여기서도 정의한다)."""


# ---------------------------------------------------------------------------
# 관측 규격(A2 소유 obs.py) 지연 접근
# ---------------------------------------------------------------------------
def _fallback_stack_frames(cur: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """계약과 같은 채널 쌓기: [..., 3] x2 → [..., 6](현재 프레임 채널이 앞)."""

    return np.concatenate([cur, prev], axis=-1).astype(np.uint8, copy=False)


def _fallback_proprio(ego_v: float | np.ndarray, domain: str) -> np.ndarray:
    return (np.asarray(ego_v, dtype=np.float32) / np.float32(_FALLBACK_SPEED[domain]))[..., None]


_FALLBACK_SPEED: dict[str, float] = {"driving": 30.0, "robot": 3.0}
_FALLBACK_ACCEL: dict[str, float] = {"driving": 4.0, "robot": 1.0}
_FALLBACK_OBS = SimpleNamespace(
    IMG_SIZE=(64, 64),
    HISTORY_OFFSET=2,
    speed_scale=lambda domain: _FALLBACK_SPEED[domain],
    accel_scale=lambda domain: _FALLBACK_ACCEL[domain],
    stack_frames=_fallback_stack_frames,
    proprio=_fallback_proprio,
)
_warned_fallback = False


def _obs() -> ModuleType | SimpleNamespace:
    """`vcp.vla.obs`(A2)를 반환한다. 없으면 계약 값과 같은 대체 구현을 반환하고 1회 경고한다."""

    global _warned_fallback
    try:
        from vcp.vla import obs  # 지연 import: A2와 병렬 개발

        return obs
    except ImportError:
        if not _warned_fallback:
            logger.warning("vcp.vla.obs를 찾지 못해 계약 값과 같은 대체 관측 규격을 사용합니다.")
            _warned_fallback = True
        return _FALLBACK_OBS


def _stack_batch(cur: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """[B,H,W,3] 두 묶음을 [B,H,W,6]으로 쌓는다. obs.stack_frames가 배치를 받지 않으면 샘플별로 쌓는다."""

    stack = _obs().stack_frames
    try:
        out = np.asarray(stack(cur, prev))
        if out.shape == cur.shape[:-1] + (cur.shape[-1] * 2,):
            return out
    except (ValueError, AssertionError, TypeError):
        pass
    return np.stack([np.asarray(stack(c, p)) for c, p in zip(cur, prev)], axis=0)


def _vocab_size(tokens: np.ndarray) -> int:
    """어휘 크기: `instructions.VOCAB`(A2)가 있으면 그 크기, 없으면 토큰 최댓값 + 1(최소 2: PAD, UNK)."""

    n = int(tokens.max()) + 1 if tokens.size else 2
    try:
        from vcp.vla.instructions import VOCAB  # 지연 import

        n = max(n, len(VOCAB))
    except ImportError:
        logger.info("vcp.vla.instructions가 없어 토큰 최댓값으로 어휘 크기(%d)를 정합니다.", n)
    return max(n, 2)


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
@dataclass
class PolicyConfig:
    """VLA-lite 구조·학습 설정(계약 필드 + 기본값이 있는 확장 필드).

    계약 필드: chunk, vis_dim, lang_dim, use_language, steps, batch, lr, weight_decay, seed, augment, threads.
    확장 필드(기본값으로 계약 동작 유지): huber_beta, front_weight, warmup_steps, grad_clip,
    n_curve_bins, device.
    """

    chunk: int = 8
    vis_dim: int = 128
    lang_dim: int = 64
    use_language: bool = True
    steps: int = 3000
    batch: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    augment: bool = True
    threads: int = 1
    # --- 확장(계약에 없는 필드, 모두 기본값 있음) ---
    huber_beta: float = 1.0          # SmoothL1 전환점(정규화 단위)
    front_weight: float = 0.5        # 청크 첫 원소 가중치 = 1+front_weight, 마지막 = 1(평균 1로 재정규화)
    warmup_steps: int = 100          # 선형 워밍업 단계(steps의 5%를 넘지 않게 자른다)
    grad_clip: float = 1.0           # 기울기 노름 상한(0 이하면 끔)
    n_curve_bins: int = 30           # 손실 곡선 구간 수
    device: str = "cpu"              # 학습 장치(배포·실험 기본은 CPU, 3080 Ti 학습 시 "cuda")
    # --- v3 검출 특징 토큰(P1, docs/33). 기본값 False면 v2 모델과 구조·결정성·파라미터 수가 같다 ---
    use_features: bool = False       # 검출 특징 [B,2,feature_dim](t, t−2)을 헤드 입력에 결합할지
    feature_dim: int = 32            # 프레임당 특징 차원(v1 16 + v2 16)
    feature_hidden: int = 64         # 특징 MLP 은닉·출력 차원
    feature_noise: float = 0.02      # 학습 증강: 특징에 더하는 가우시안 노이즈 σ(augment=True일 때만, 0이면 끔)
    # --- v4(docs/36). 기본값이면 v3와 구조·초기 가중치·학습 결과가 비트 단위로 같다 ---
    feature_history: int = 2         # P2: 특징 이력 프레임 수 H(인덱스 k = 프레임 t − k·stride, k=0이 현재)
    feature_stride: int = 2          # P2: 특징 이력 간격 s(프레임)
    feature_encoder: str = "mlp"     # P2: "mlp"(H프레임 평탄화 MLP, H=2이면 v3) | "gru"(Linear+ReLU → GRU 1층)
    aux_weight: float = 0.0          # P3: 위험 맥락 보조 헤드 손실 가중치 α(0이면 보조 헤드 없음)
    aux_classes: int = 6             # P3: 맥락 클래스 수(PolicyData.aux_targets [N, aux_classes])
    lang_dropout: float = 0.0        # T1: 학습 중 표본별로 지시문을 빈 지시(PAD만)로 바꿀 확률 p
    proprio_history: int = 1         # P5: 자차 속도 이력 프레임 수 Hp(1이면 현재 속도만, v3와 같음)
    proprio_stride: int = 2          # P5: 속도 이력 간격(프레임)
    goal_encoding: bool = False      # P6: 지시문의 목표 속도 숫자를 결정적으로 읽어 [g, 있음, g − v_t]를 입력(언어 사용 시만)
    goal_residual_gain: float = 0.0  # P7: 잔차 행동 기준 a0 = clip(gain·(g − v_t), ±goal_residual_clip)(정규화 단위). 0이면 끔
    goal_residual_clip: float = 0.5  # P7: 기준 행동 상한(정규화 가속도 단위)
    lang_embedding: bool = True      # 절제용: False면 학습 언어 임베딩·FiLM 경로를 끄고(0 벡터) 토큰은 P6·P7 목표 읽기에만 쓴다

    def __post_init__(self) -> None:
        """v4 확장 필드 검사(잘못된 값은 학습 전에 바로 알린다)."""

        if self.feature_encoder not in FEATURE_ENCODERS:
            raise ValueError(f"feature_encoder는 {FEATURE_ENCODERS} 중 하나여야 합니다: {self.feature_encoder!r}")
        if int(self.feature_history) < 1 or int(self.feature_stride) < 1:
            raise ValueError(
                f"feature_history({self.feature_history}), feature_stride({self.feature_stride})는 1 이상이어야 합니다."
            )
        if not (self.aux_weight >= 0.0 and math.isfinite(self.aux_weight)):
            raise ValueError(f"aux_weight는 0 이상의 유한한 값이어야 합니다: {self.aux_weight}")
        if self.aux_weight > 0 and int(self.aux_classes) < 2:
            raise ValueError(f"aux_weight > 0이면 aux_classes는 2 이상이어야 합니다: {self.aux_classes}")
        if not (0.0 <= self.lang_dropout <= 1.0):
            raise ValueError(f"lang_dropout은 [0, 1] 범위여야 합니다: {self.lang_dropout}")
        if int(self.proprio_history) < 1 or int(self.proprio_stride) < 1:
            raise ValueError(f"proprio_history({self.proprio_history}), proprio_stride({self.proprio_stride})는 1 이상이어야 합니다.")


FEATURE_ENCODERS: tuple[str, ...] = ("mlp", "gru")
"""P2 시간 특징 인코더 종류. "mlp"는 v3 방식(H=2이면 v3와 같음), "gru"는 v4 시간 인코더."""


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
class _ConvBlock(nn.Module):
    """Conv(k×k, stride s) → GroupNorm → [FiLM] → ReLU."""

    def __init__(self, c_in: int, c_out: int, kernel: int = 3, stride: int = 2, groups: int = 8) -> None:
        super().__init__()
        padding = (kernel - 1) // 2 if kernel % 2 == 1 else 0
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.norm = nn.GroupNorm(min(groups, c_out), c_out)

    def forward(self, x: torch.Tensor, film: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        x = self.norm(self.conv(x))
        if film is not None:
            gamma, beta = film
            x = x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return F.relu(x)


class VLALitePolicy(nn.Module):
    """소형 VLA 정책: 6채널 영상 CNN + FiLM 언어 조건화 + proprio → 정규화 가속도 청크 [B, chunk].

    구조
    - 시각: 입력 [B,6,64,64](t와 t−2 RGB, [0,1]) → (x−0.5)/0.25 → Conv 3단(GroupNorm, ReLU)
      - 줄기(stem) 4×4 stride 4(겹치지 않는 패치 투영, 32ch, 16×16)
      - 3×3 stride 2(64ch, 8×8) → 3×3 stride 2(128ch, 4×4) → flatten → Linear → vis_dim.
      처음 검토한 3×3 stride-2 4단(32-64-96-128)은 CPU 1스레드에서 64×64 해상도의 앞단 conv와
      GroupNorm 비용이 커서(배치 128 순전파+역전파 약 350 ms) 3000단계 학습이 20분을 넘었다.
      패치 줄기는 모든 픽셀을 보면서(겹치지 않는 4×4 패치의 선형 투영) 앞단 연산을 약 1/4로 줄인다.
      메모리 배치는 channels_last를 쓴다(CPU oneDNN conv가 더 빠르다).
    - 풀링 선택: 전역 평균 대신 **4×4 격자를 그대로 flatten + Linear**한다. 주행 정책에서는
      선행 객체가 "자기 차선(화면 중앙 하단)에 있는가"와 "얼마나 큰가(거리)"가 핵심인데, 전역 평균은
      위치를 지우고, 공간 softmax는 채널별 기대 좌표만 남겨 크기·존재 강도 정보를 잃는다.
      4×4 flatten은 위치와 크기를 모두 보존하면서 파라미터가 작다(2048×vis_dim).
      입력 크기가 달라도 동작하도록 flatten 전에 AdaptiveAvgPool2d(4)를 둔다(64×64에서는 이미 4×4라 건너뛴다).
    - 언어: 토큰 임베딩(lang_dim, PAD=0) → 마스크 평균 → Linear+ReLU → lang_dim.
      지시문은 스타일·목표 속도 단어가 핵심인 짧은 템플릿 문장이므로 어순보다 단어 존재가 중요해
      GRU 대신 마스크 평균(bag-of-words)을 쓴다(CPU 처리량 우선).
    - FiLM: 언어 특징 → (γ, β)로 `film_blocks` 블록(기본 줄기 다음의 2·3번째 conv 블록)의 GroupNorm 출력을
      x·(1+γ)+β로 조건화한다(RT-1의 FiLM-EfficientNet과 같은 발상). 생성기 가중치를 0으로 초기화해
      학습 시작 시 항등 변환이 되게 한다.
    - proprio: 정규화 속도 [B,1] → MLP(32) → 32.
    - 헤드: concat(vis, lang, proprio) → MLP(256, 256) → chunk.

    `use_language=False`이면 언어 모듈을 만들지 않고, FiLM은 항등, 헤드의 언어 입력은 0 벡터인 절제 모델이 된다.

    v3 검출 특징 토큰(`use_features=True`, docs/33 P1)
    - 입력 features float [B,2,feature_dim](프레임 t, t−2의 v1+v2 특징) → clamp(−3,3)(NaN은 0) → 평탄화 [B,2D]
      → MLP(2D → feature_hidden → feature_hidden, ReLU) → 헤드 입력 concat(vis, lang, proprio, feat).
    - 특징은 대부분 [−1,1] 근처라 별도 정규화는 두지 않고 이상치만 자른다.
    - 특징 모듈은 다른 모든 모듈을 만든 **뒤에** 만든다. use_features=False에서는 만들지 않으므로
      기존 모델과 파라미터 수·초기화 난수 소비·순전파가 완전히 같다.

    v4 블록(docs/36, 기본값이면 v3와 같은 모듈·같은 생성 순서)
    - P2 시간 특징 인코더: 입력 [B, feature_history=H, D](k=0이 현재 프레임 t, k는 t−k·stride).
      - `feature_encoder="mlp"`: [B,H·D] 평탄화 → MLP(H·D → hidden → hidden). H=2이면 v3 모듈과 같다.
      - `feature_encoder="gru"`: NaN→0, clamp(−3,3) → 시간 순서(오래된 것 → 현재)로 뒤집음 → Linear(D→hidden)+ReLU
        → GRU(hidden→hidden, 1층) → 마지막 은닉 [B,hidden]. 출력 차원이 MLP와 같아 헤드 입력 차원이 유지된다.
    - P3 위험 맥락 보조 헤드(`aux_classes > 0`): 헤드 첫 층 입력(융합 표현)에서 Linear(→aux_classes) 로짓.
      학습 손실에만 쓰고 `forward`의 기본 반환(행동 청크)에는 영향이 없다(`return_aux=True`일 때만 계산).
    - 생성 순서: 기존 모듈 → 특징 인코더(mlp 또는 gru) → 보조 헤드. 새 모듈이 늘 마지막이라
      기존 모듈의 초기화 난수 소비가 바뀌지 않는다.
    """

    def __init__(
        self,
        vocab_size: int,
        chunk: int = 8,
        vis_dim: int = 128,
        lang_dim: int = 64,
        use_language: bool = True,
        widths: tuple[int, ...] = (32, 64, 128),
        kernels: tuple[int, ...] = (4, 3, 3),
        strides: tuple[int, ...] = (4, 2, 2),
        film_blocks: tuple[int, ...] = (1, 2),
        in_channels: int = 6,
        proprio_dim: int = 1,
        proprio_hidden: int = 32,
        head_hidden: int = 256,
        grid: int = 4,
        use_features: bool = False,
        feature_dim: int = 32,
        feature_hidden: int = 64,
        feature_history: int = 2,
        feature_stride: int = 2,
        feature_encoder: str = "mlp",
        aux_classes: int = 0,
        proprio_stride: int = 2,
        goal_encoding: bool = False,
        goal_residual_gain: float = 0.0,
        goal_residual_clip: float = 0.5,
        lang_embedding: bool = True,
    ) -> None:
        super().__init__()
        if feature_encoder not in FEATURE_ENCODERS:
            raise ValueError(f"feature_encoder는 {FEATURE_ENCODERS} 중 하나여야 합니다: {feature_encoder!r}")
        if feature_history < 1 or feature_stride < 1:
            raise ValueError(f"feature_history({feature_history}), feature_stride({feature_stride})는 1 이상이어야 합니다.")
        if aux_classes < 0:
            raise ValueError(f"aux_classes는 0 이상이어야 합니다(0 = 보조 헤드 없음): {aux_classes}")
        if vocab_size < 2:
            raise ValueError("vocab_size는 2 이상이어야 합니다(PAD=0, UNK=1).")
        if not (len(widths) == len(kernels) == len(strides)):
            raise ValueError("widths, kernels, strides 길이가 같아야 합니다.")
        if any(b < 0 or b >= len(widths) for b in film_blocks):
            raise ValueError(f"film_blocks {film_blocks}가 conv 블록 수 {len(widths)}를 벗어납니다.")
        self.vocab_size = int(vocab_size)
        self.chunk = int(chunk)
        self.lang_dim = int(lang_dim)
        self.use_language = bool(use_language)
        self.film_blocks = tuple(film_blocks)
        if use_features and (feature_dim < 1 or feature_hidden < 1):
            raise ValueError(f"feature_dim({feature_dim}), feature_hidden({feature_hidden})은 1 이상이어야 합니다.")
        self.use_features = bool(use_features)
        self.feature_dim = int(feature_dim)
        self.feature_hidden = int(feature_hidden) if self.use_features else 0
        # 특징 이력 규칙(관측 생성용 메타데이터). 모델 계산에는 history만 쓰고 stride는 관측 생성(predict_open_loop,
        # 폐루프)이 학습과 같은 규칙을 쓰도록 함께 보관한다.
        self.feature_history = int(feature_history)
        self.feature_stride = int(feature_stride)
        self.feature_encoder = str(feature_encoder)
        self.aux_classes = int(aux_classes)
        # P5 자차 운동 이력: proprio_dim = Hp(이력 프레임 수). 관측 생성 규칙(stride)도 함께 보관한다.
        self.proprio_history = int(proprio_dim)
        self.proprio_stride = int(proprio_stride)
        # P6 수치 목표 인코딩: 언어를 쓰는 모델에서만 켠다(언어 절제 모델은 목표도 받지 않는다)
        self.goal_encoding = bool(goal_encoding) and self.use_language
        self.goal_hidden = 16 if self.goal_encoding else 0
        # P7 목표 속도 잔차 헤드: 목표 표가 필요하므로 P6(goal_encoding)이 켜진 모델에서만 쓴다. 학습 파라미터는 없다.
        self.goal_residual_gain = float(goal_residual_gain) if self.goal_encoding else 0.0
        self.goal_residual_clip = float(goal_residual_clip)
        # 절제(3차 심사 M2): 학습 언어 경로만 끄고 수치 목표(P6·P7)는 남긴다. 모듈은 그대로 만들어 난수 순서를 바꾸지 않는다
        self.lang_embedding = bool(lang_embedding)

        chans = (in_channels,) + tuple(widths)
        self.blocks = nn.ModuleList(
            _ConvBlock(chans[i], chans[i + 1], kernel=kernels[i], stride=strides[i]) for i in range(len(widths))
        )
        self.grid = int(grid)
        self.pool = nn.AdaptiveAvgPool2d(grid)
        self.vis_fc = nn.Linear(widths[-1] * grid * grid, vis_dim)

        if self.use_language:
            self.embed = nn.Embedding(self.vocab_size, lang_dim, padding_idx=0)
            self.lang_fc = nn.Linear(lang_dim, lang_dim)
            self.film = nn.ModuleDict({str(b): nn.Linear(lang_dim, 2 * widths[b]) for b in self.film_blocks})
            for lin in self.film.values():
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)

        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, proprio_hidden), nn.ReLU(), nn.Linear(proprio_hidden, proprio_hidden), nn.ReLU()
        )
        self.fused_dim = vis_dim + lang_dim + proprio_hidden + self.feature_hidden + self.goal_hidden
        self.head = nn.Sequential(
            nn.Linear(self.fused_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, chunk),
        )
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)
        # 아래 모듈은 마지막에 만든다(기존 모듈의 초기화 난수 소비 순서를 바꾸지 않기 위해). 순서: 특징 인코더 → 보조 헤드.
        if self.use_features:
            if self.feature_encoder == "mlp":
                self.feat_mlp = nn.Sequential(
                    nn.Linear(self.feature_history * self.feature_dim, self.feature_hidden),
                    nn.ReLU(),
                    nn.Linear(self.feature_hidden, self.feature_hidden),
                    nn.ReLU(),
                )
            else:  # "gru"
                self.feat_in = nn.Linear(self.feature_dim, self.feature_hidden)
                self.feat_gru = nn.GRU(self.feature_hidden, self.feature_hidden, num_layers=1, batch_first=True)
        if self.aux_classes > 0:
            self.aux_head = nn.Linear(self.fused_dim, self.aux_classes)
        if self.goal_encoding:  # P6(맨 마지막에 만든다): 목표 값 표는 버퍼(난수 소비 없음), MLP 3 → 16 → 16
            from .instructions import goal_value_table

            table = torch.from_numpy(goal_value_table())
            if len(table) < self.vocab_size:
                table = torch.cat([table, torch.full((self.vocab_size - len(table),), float("nan"))])
            self.register_buffer("goal_table", table[: self.vocab_size].clone(), persistent=True)
            self.goal_mlp = nn.Sequential(nn.Linear(3, self.goal_hidden), nn.ReLU(), nn.Linear(self.goal_hidden, self.goal_hidden), nn.ReLU())
        self.to(memory_format=torch.channels_last)

    @classmethod
    def from_config(cls, cfg: PolicyConfig, vocab_size: int) -> "VLALitePolicy":
        return cls(
            vocab_size,
            chunk=cfg.chunk,
            vis_dim=cfg.vis_dim,
            lang_dim=cfg.lang_dim,
            use_language=cfg.use_language,
            use_features=cfg.use_features,
            feature_dim=cfg.feature_dim,
            feature_hidden=cfg.feature_hidden,
            feature_history=cfg.feature_history,
            feature_stride=cfg.feature_stride,
            feature_encoder=cfg.feature_encoder,
            aux_classes=cfg.aux_classes if cfg.aux_weight > 0 else 0,
            proprio_dim=cfg.proprio_history,
            proprio_stride=cfg.proprio_stride,
            goal_encoding=cfg.goal_encoding,
            goal_residual_gain=cfg.goal_residual_gain,
            goal_residual_clip=cfg.goal_residual_clip,
            lang_embedding=cfg.lang_embedding,
        )

    def encode_features(self, features: torch.Tensor | None) -> torch.Tensor:
        """특징 [B,H,feature_dim](H = feature_history, k=0이 현재 프레임) → [B,feature_hidden].

        NaN은 0, 값은 [−3,3]으로 자른 뒤 인코더에 넣는다.
        - "mlp": 평탄화 [B,H·D] → MLP. H=2이면 v3와 같은 계산이다.
        - "gru": 시간 축을 뒤집어(오래된 것 → 현재) Linear+ReLU → GRU 1층 → 마지막 은닉 [B,hidden].

        Raises:
            ValueError: use_features=True인데 features가 없거나 형태가 [B,feature_history,feature_dim]이 아닐 때.
        """

        h, d = self.feature_history, self.feature_dim
        if features is None:
            raise ValueError(f"use_features=True 모델에는 features [B,{h},{d}] 입력이 필요합니다.")
        if features.ndim != 3 or features.shape[1] != h or features.shape[2] != d:
            raise ValueError(f"features 형태가 [B,{h},{d}]이 아닙니다: {tuple(features.shape)}")
        f = torch.nan_to_num(features.float(), nan=0.0).clamp(-3.0, 3.0)
        if self.feature_encoder == "mlp":
            return self.feat_mlp(f.flatten(1))
        x = F.relu(self.feat_in(f.flip(1)))  # [B,H,hidden], 시간 순서(오래된 것 → 현재 프레임)
        _, h_n = self.feat_gru(x)             # h_n [1,B,hidden]
        return h_n[-1]

    def prep_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        """P5 전처리: [B,Hp] 정규화 속도 이력(k=0 현재) → [v_t, 10·(v_t − v_{t−s}), 10·(v_{t−s} − v_{t−2s}), …].

        Hp=1이면 입력을 그대로 돌려준다(v3와 같음). 차분은 정규화 속도 단위가 작아 10배로 키운다.

        Raises:
            ValueError: 입력 형태가 [B, proprio_history]가 아닐 때.
        """

        if proprio.ndim != 2 or proprio.shape[1] != self.proprio_history:
            raise ValueError(f"proprio 형태가 [B,{self.proprio_history}]가 아닙니다: {tuple(proprio.shape)}")
        if self.proprio_history == 1:
            return proprio
        diff = (proprio[:, :-1] - proprio[:, 1:]) * 10.0
        return torch.cat([proprio[:, :1], diff], dim=1)

    def fuse(
        self,
        image: torch.Tensor,
        tokens: torch.Tensor,
        proprio: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """융합 표현(헤드 첫 층 입력) [B, fused_dim] = concat(vis, lang, proprio[, feat])."""

        lang = self.encode_language(tokens)
        x = image.mul(4.0).sub_(2.0).contiguous(memory_format=torch.channels_last)  # (x−0.5)/0.25
        for i, block in enumerate(self.blocks):
            film = None
            if self.use_language and i in self.film_blocks:
                gamma, beta = self.film[str(i)](lang).chunk(2, dim=-1)
                film = (gamma, beta)
            x = block(x, film)
        if x.shape[-2:] != (self.grid, self.grid):  # 64×64 입력에서는 이미 grid×grid라 풀링을 건너뛴다(CPU 역전파 비용 절감)
            x = self.pool(x)
        vis = F.relu(self.vis_fc(x.flatten(1)))
        prop = self.proprio_mlp(self.prep_proprio(proprio))
        parts = [vis, lang, prop]
        if self.use_features:
            parts.append(self.encode_features(features))
        if self.goal_encoding:
            parts.append(self.goal_mlp(self.encode_goal(tokens, proprio[:, :1])))
        return torch.cat(parts, dim=-1)

    def goal_base_action(self, tokens: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """P7 기준 행동 [B,1] = clip(gain·(g − v_t), ±clip) × 있음. 지시문에 목표 숫자가 없으면 0."""

        g, has, _ = self.encode_goal(tokens, proprio[:, :1]).unbind(1)
        base = (self.goal_residual_gain * (g - proprio[:, 0])).clamp(-self.goal_residual_clip, self.goal_residual_clip) * has
        return base[:, None]

    def encode_goal(self, tokens: torch.Tensor, v_now: torch.Tensor) -> torch.Tensor:
        """P6: 토큰 [B,L] → [g, 있음(0/1), g − v_t] [B,3]. g는 지시문 숫자 토큰의 정규화 목표 속도(없으면 0).

        숫자 토큰이 여러 개면 평균한다. 빈 지시(지시문 드롭아웃)나 숫자 없는 지시문은 (0, 0, 0)이다.
        """

        tok = torch.where(tokens >= self.vocab_size, torch.ones_like(tokens), tokens).clamp_min(0)
        vals = self.goal_table[tok]
        ok = torch.isfinite(vals)
        cnt = ok.sum(1, keepdim=True).float()
        has = (cnt > 0).float()
        g = torch.where(ok, vals, torch.zeros_like(vals)).sum(1, keepdim=True) / cnt.clamp_min(1.0)
        return torch.cat([g, has, (g - v_now) * has], dim=1)

    def encode_language(self, tokens: torch.Tensor) -> torch.Tensor:
        """토큰 [B,L] → 언어 특징 [B,lang_dim]. 어휘 밖 id는 UNK(1)로 바꾼다. 절제 모델은 0 벡터."""

        b = tokens.shape[0]
        if not self.use_language or not self.lang_embedding:
            return torch.zeros(b, self.lang_dim, device=tokens.device)
        tok = torch.where(tokens >= self.vocab_size, torch.ones_like(tokens), tokens).clamp_min(0)
        mask = (tok != 0).float()
        emb = self.embed(tok) * mask[..., None]
        mean = emb.sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)
        return F.relu(self.lang_fc(mean))

    def forward(
        self,
        image: torch.Tensor,
        tokens: torch.Tensor,
        proprio: torch.Tensor,
        features: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """image float [B,6,H,W]([0,1]), tokens int64 [B,L], proprio float [B,1] → 정규화 가속도 [B,chunk].

        features float [B,feature_history,feature_dim]은 use_features=True일 때만 쓰고(필수), False면 무시한다.
        `return_aux=True`이면 (행동 [B,chunk], 보조 로짓 [B,aux_classes])를 반환한다. 기본(False)은 행동만
        반환하며 보조 헤드를 계산하지 않는다(추론 비용 0).

        Raises:
            ValueError: return_aux=True인데 보조 헤드가 없을 때(aux_classes=0).
        """

        if return_aux and self.aux_classes <= 0:
            raise ValueError("보조 헤드가 없는 모델입니다(aux_classes=0, PolicyConfig.aux_weight=0).")
        fused = self.fuse(image, tokens, proprio, features)
        action = self.head(fused)
        if self.goal_residual_gain > 0:  # P7: 정책 출력 = 해석적 목표 속도 추종 기준 + 학습 잔차
            action = action + self.goal_base_action(tokens, proprio)
        if return_aux:
            return action, self.aux_head(fused)
        return action


def count_parameters(model: nn.Module) -> int:
    """학습 가능한 파라미터 수."""

    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ---------------------------------------------------------------------------
# 영상 소스
# ---------------------------------------------------------------------------
@runtime_checkable
class ImageSource(Protocol):
    """프레임 인덱스 배열 → uint8 [B,H,W,3] RGB 프레임."""

    def get(self, idx: np.ndarray) -> np.ndarray: ...


class ArrayImageSource:
    """미리 만든 프레임 배열(예: comma 64×64 `frames`)을 그대로 돌려주는 소스."""

    def __init__(self, frames: np.ndarray) -> None:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"frames는 [N,H,W,3]이어야 합니다: {frames.shape}")
        self.frames = frames if frames.dtype == np.uint8 else frames.astype(np.uint8)

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def get(self, idx: np.ndarray) -> np.ndarray:
        return self.frames[np.asarray(idx, dtype=np.int64)]


class PoolImageSource:
    """pool의 GT 투영 박스(`box_ptr`, `boxes`)와 `horizon_y`로 프레임을 지연 렌더링한다.

    - 렌더러는 `vcp.sim.render.render_frame`(A2)이며 첫 렌더링 때 지연 import한다.
    - 렌더 결과는 LRU 캐시에 프레임 단위로 보관한다. `max_cache` 프레임을 넘으면 가장 오래 안 쓴 것부터
      버린다(64×64×3 = 12 KB/프레임, 기본 100,000 프레임 ≈ 1.2 GB). 0이면 캐시하지 않는다.
    """

    def __init__(
        self,
        box_ptr: np.ndarray,
        boxes: np.ndarray,
        horizon_y: np.ndarray,
        domain: str = "driving",
        size: tuple[int, int] | None = None,
        max_cache: int = 100_000,
    ) -> None:
        self.box_ptr = np.asarray(box_ptr, dtype=np.int64)
        self.boxes = np.asarray(boxes, dtype=np.float32)
        self.horizon_y = np.asarray(horizon_y, dtype=np.float32)
        if len(self.box_ptr) != len(self.horizon_y) + 1:
            raise ValueError("box_ptr 길이는 프레임 수 + 1이어야 합니다.")
        self.domain = domain
        self.size = tuple(size) if size is not None else tuple(_obs().IMG_SIZE)
        self.max_cache = int(max_cache)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._render: Callable[..., np.ndarray] | None = None
        self.hits = 0
        self.misses = 0

    @classmethod
    def from_pool(cls, arrays: dict[str, np.ndarray], domain: str, **kwargs: Any) -> "PoolImageSource":
        """`load_pool`이 돌려준 배열 dict에서 만든다."""

        return cls(arrays["box_ptr"], arrays["boxes"], arrays["horizon_y"], domain=domain, **kwargs)

    def __len__(self) -> int:
        return int(len(self.horizon_y))

    def _render_one(self, i: int) -> np.ndarray:
        if self._render is None:
            from vcp.sim.render import render_frame  # 지연 import: A2와 병렬 개발

            self._render = render_frame
        b = self.boxes[self.box_ptr[i] : self.box_ptr[i + 1]]
        return np.asarray(self._render(b, float(self.horizon_y[i]), domain=self.domain, size=self.size), dtype=np.uint8)

    def get(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64).ravel()
        h, w = self.size
        out = np.empty((len(idx), h, w, 3), dtype=np.uint8)
        cache = self._cache
        for k, i in enumerate(idx.tolist()):
            frame = cache.get(i)
            if frame is None:
                frame = self._render_one(i)
                self.misses += 1
                if self.max_cache > 0:
                    cache[i] = frame
                    if len(cache) > self.max_cache:
                        cache.popitem(last=False)
            else:
                cache.move_to_end(i)
                self.hits += 1
            out[k] = frame
        return out

    def cache_info(self) -> dict[str, int]:
        return {"size": len(self._cache), "max_cache": self.max_cache, "hits": self.hits, "misses": self.misses}


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------
def _run_bounds(group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """각 프레임이 속한 연속 그룹 구간의 첫·마지막 인덱스를 [N] 배열 2개로 반환한다."""

    n = len(group)
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    change = np.flatnonzero(group[1:] != group[:-1]) + 1
    starts = np.concatenate([[0], change]).astype(np.int64)
    ends = np.concatenate([change - 1, [n - 1]]).astype(np.int64)
    lengths = ends - starts + 1
    return np.repeat(starts, lengths), np.repeat(ends, lengths)


@dataclass
class PolicyData:
    """정책 학습·평가용 프레임 정렬 데이터(N = 전체 프레임).

    - `images`: 프레임 소스(ImageSource), `group` [N]: 에피소드·블록 id(같은 값이 **연속 구간**을 이뤄야 한다)
    - `ego_v` [N]: 자차 속도(m/s), `action` [N]: 물리 단위 가속도 행동(m/s²)
    - `tokens` [N,L] int64: 지시문 토큰, `domain`: "driving" | "robot"
    - `features` [N,D] float32 | None(v3): 프레임별 검출 특징(v1+v2, 보통 D=32). 있으면 관측에 `"features"`를 넣는다.
    - `aux_targets` [N,C] float32 | None(v4 P3): 프레임별 맥락 확률(CARE 점수기 출력, 각 행의 합 1, 음수 없음).
      보조 헤드 학습 타깃(현재 프레임 t의 행)이며 관측 dict에는 넣지 않는다(추론 입력이 아님).
    - `feature_group` [N] | None(v4 특징 프리롤): 특징 이력의 잘라 냄 기준 그룹(보통 에피소드 id, 같은 값이 연속 구간).
      None이면 `group`을 쓴다(v3와 같음). 학습 데이터의 `group`이 클립 id여도 `feature_group`=에피소드 id를 주면
      특징 이력이 클립 시작 이전(같은 에피소드의 앞 프레임 = 특징 프리롤)까지 이어져 폐루프(에피소드 시작에서
      잘라 냄)와 같은 규칙이 된다. 영상·행동 청크는 계속 `group` 기준이다. 이 경우 `features`는 프리롤 프레임을
      포함한 전체 길이 N 배열이어야 하고, 학습 인덱스(train_idx)는 클립 프레임만 고르면 된다.

    경계 규칙
    - 관측 = stack_frames(t, t−HISTORY_OFFSET). t−2가 같은 연속 구간의 시작보다 앞이면 구간 첫 프레임을 쓴다.
    - 특징 이력(v4) = features[t − k·stride], k = 0..history−1 → [B,history,D]. 같은 연속 구간(`feature_group`이
      있으면 그 구간, 없으면 `group` 구간) 시작보다 앞이면 구간 첫 프레임으로 잘라 낸다(영상 `prev_index`와 같은
      규칙). history=2, stride=2, feature_group=None(기본)이면 v3의 [features[t], features[t−2]]와 같은 배열이다.
    - 행동 청크 = action[t : t+chunk], 구간 끝을 넘는 자리는 구간 마지막 값으로 채운다.
    """

    images: ImageSource
    group: np.ndarray
    ego_v: np.ndarray
    action: np.ndarray
    tokens: np.ndarray
    domain: str
    features: np.ndarray | None = None
    aux_targets: np.ndarray | None = None
    feature_group: np.ndarray | None = None
    loss_weight: np.ndarray | None = None  # v4 P4: 프레임별 행동 손실 가중치 [N](양수). None이면 균등(v3와 같음)
    _first: np.ndarray = field(init=False, repr=False)
    _feat_first: np.ndarray = field(init=False, repr=False)
    _last: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.group = np.asarray(self.group)
        self.ego_v = np.asarray(self.ego_v, dtype=np.float32)
        self.action = np.asarray(self.action, dtype=np.float32)
        self.tokens = np.asarray(self.tokens, dtype=np.int64)
        n = len(self.group)
        if not (len(self.ego_v) == len(self.action) == len(self.tokens) == n):
            raise ValueError(
                f"길이 불일치: group={n}, ego_v={len(self.ego_v)}, action={len(self.action)}, tokens={len(self.tokens)}"
            )
        if self.tokens.ndim != 2:
            raise ValueError(f"tokens는 [N,L]이어야 합니다: {self.tokens.shape}")
        if self.features is not None:
            self.features = np.asarray(self.features, dtype=np.float32)
            if self.features.ndim != 2 or len(self.features) != n:
                raise ValueError(f"features는 [N={n}, D]이어야 합니다: {self.features.shape}")
            if not np.isfinite(self.features).all():
                logger.warning("features에 유한하지 않은 값이 있습니다. 모델 입력에서 NaN은 0, ±inf는 ±3으로 처리됩니다.")
        if self.aux_targets is not None:
            self.aux_targets = np.asarray(self.aux_targets, dtype=np.float32)
            q = self.aux_targets
            if q.ndim != 2 or len(q) != n or q.shape[1] < 2:
                raise ValueError(f"aux_targets는 [N={n}, C≥2]이어야 합니다: {q.shape}")
            if not np.isfinite(q).all() or (q < 0).any():
                raise ValueError("aux_targets는 유한하고 음수가 없는 확률이어야 합니다.")
            dev = float(np.abs(q.sum(axis=1) - 1.0).max()) if n else 0.0
            if dev > 1e-3:
                raise ValueError(f"aux_targets 각 행의 합이 1이어야 합니다(최대 편차 {dev:.4g}).")
        if self.loss_weight is not None:
            self.loss_weight = np.asarray(self.loss_weight, dtype=np.float32)
            if self.loss_weight.shape != (n,) or not np.isfinite(self.loss_weight).all() or (self.loss_weight <= 0).any():
                raise ValueError(f"loss_weight는 양수·유한한 [N={n}] 배열이어야 합니다: {self.loss_weight.shape}")
        self._first, self._last = _run_bounds(self.group)
        if self.feature_group is not None:
            self.feature_group = np.asarray(self.feature_group)
            if self.feature_group.ndim != 1 or len(self.feature_group) != n:
                raise ValueError(f"feature_group은 [N={n}]이어야 합니다: {self.feature_group.shape}")
            self._feat_first = _run_bounds(self.feature_group)[0]
        else:
            self._feat_first = self._first

    def __len__(self) -> int:
        return int(len(self.group))

    def prev_index(self, idx: np.ndarray) -> np.ndarray:
        """t−HISTORY_OFFSET 인덱스(같은 그룹 구간 시작보다 앞이면 구간 첫 프레임)."""

        idx = np.asarray(idx, dtype=np.int64)
        return np.maximum(idx - int(_obs().HISTORY_OFFSET), self._first[idx])

    def feature_index(self, idx: np.ndarray, history: int = 2, stride: int = 2) -> np.ndarray:
        """[len(idx), history] 특징 이력 인덱스. 열 k = t − k·stride.

        같은 구간 시작보다 앞이면 구간 첫 프레임으로 잘라 낸다. 구간은 `feature_group`이 있으면 그 기준
        (특징 프리롤 허용), 없으면 `group` 기준이다.

        Raises:
            ValueError: history < 1 또는 stride < 1일 때.
        """

        if history < 1 or stride < 1:
            raise ValueError(f"history({history}), stride({stride})는 1 이상이어야 합니다.")
        idx = np.asarray(idx, dtype=np.int64)
        offs = np.arange(history, dtype=np.int64) * int(stride)
        return np.maximum(idx[:, None] - offs[None, :], self._feat_first[idx][:, None])

    def chunk_index(self, idx: np.ndarray, chunk: int) -> np.ndarray:
        """[len(idx), chunk] 행동 인덱스. 구간 끝을 넘으면 구간 마지막 인덱스로 채운다."""

        idx = np.asarray(idx, dtype=np.int64)
        return np.minimum(idx[:, None] + np.arange(chunk, dtype=np.int64)[None, :], self._last[idx][:, None])

    def action_chunk(self, idx: np.ndarray, chunk: int) -> np.ndarray:
        """물리 단위 행동 청크 float32 [len(idx), chunk]."""

        return self.action[self.chunk_index(idx, chunk)]

    def observation(
        self, idx: np.ndarray, history: int = 2, stride: int = 2, proprio_history: int = 1, proprio_stride: int = 2
    ) -> dict[str, np.ndarray]:
        """계약 PolicyFn 입력과 같은 형식의 배치 관측 dict.

        키: "image" uint8 [B,H,W,6](프레임 t, t−HISTORY_OFFSET; history·stride와 무관), "tokens" int64 [B,L],
        "proprio" float32 [B,1], 그리고 `features`가 있으면 "features" float32 [B,history,D]
        (k번 = 프레임 t − k·stride, 구간 시작에서 잘라 냄. 기본값이면 v3의 [B,2,D]와 같은 배열).
        """

        idx = np.asarray(idx, dtype=np.int64)
        prev = self.prev_index(idx)
        frames = self.images.get(np.concatenate([idx, prev]))
        b = len(idx)
        image = _stack_batch(frames[:b], frames[b:])
        if proprio_history == 1:
            prop = np.asarray(_obs().proprio(self.ego_v[idx], self.domain), dtype=np.float32).reshape(b, 1)
        else:  # P5: [B,Hp] 속도 이력(특징 이력과 같은 잘라 냄 규칙, feature_group이 있으면 그 기준)
            pidx = self.feature_index(idx, proprio_history, proprio_stride)
            prop = np.asarray(_obs().proprio(self.ego_v[pidx], self.domain), dtype=np.float32).reshape(b, proprio_history)
        out = {"image": image, "tokens": self.tokens[idx], "proprio": prop}
        if self.features is not None:
            out["features"] = self.features[self.feature_index(idx, history, stride)]
        return out


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------
def chunk_weights(chunk: int, front_weight: float = 0.5) -> np.ndarray:
    """청크 위치별 손실 가중치: 1+front_weight(첫 원소)에서 1(마지막)로 선형 감소, 평균 1로 정규화.

    폐루프에서는 청크 첫 원소만 실행하므로 앞쪽 예측을 조금 더 중시한다. 뒤쪽 원소는 미래 의도를
    학습시키는 보조 신호로 남긴다.
    """

    if chunk == 1:
        return np.ones(1, dtype=np.float32)
    w = 1.0 + front_weight * (1.0 - np.arange(chunk, dtype=np.float32) / (chunk - 1))
    return (w / w.mean()).astype(np.float32)


def _to_tensor_batch(obs: dict[str, np.ndarray], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """관측 dict → (image float [B,6,H,W] [0,1], tokens int64, proprio float)."""

    # NHWC uint8을 permute하면 메모리상 channels_last NCHW가 된다(모델의 conv 배치와 같다).
    image = torch.from_numpy(np.ascontiguousarray(obs["image"])).to(device).permute(0, 3, 1, 2).float().mul_(1.0 / 255.0)
    tokens = torch.from_numpy(np.asarray(obs["tokens"], dtype=np.int64)).to(device)
    p = np.asarray(obs["proprio"], dtype=np.float32)
    prop = torch.from_numpy(p.reshape(len(p), -1) if p.ndim >= 2 else p.reshape(-1, 1)).to(device)
    return image, tokens, prop


def _features_tensor(obs: dict[str, np.ndarray], model: "VLALitePolicy", device: torch.device) -> torch.Tensor | None:
    """모델이 특징을 쓰면 obs["features"] → float [B,H,D] 텐서, 쓰지 않으면 None(키가 있어도 무시).

    형태 검사([B, feature_history, feature_dim])는 `VLALitePolicy.encode_features`가 한다.

    Raises:
        ValueError: use_features=True 모델인데 관측에 "features"가 없을 때.
    """

    if not model.use_features:
        return None
    if "features" not in obs or obs["features"] is None:
        raise ValueError(
            "use_features=True 정책에는 관측 dict의 'features' float32 [B,%d,%d]가 필요합니다"
            "(PolicyData.features 또는 폐루프 온라인 특징 계산을 확인하세요)." % (model.feature_history, model.feature_dim)
        )
    return torch.from_numpy(np.ascontiguousarray(obs["features"], dtype=np.float32)).to(device)


def apply_lang_dropout(tokens: np.ndarray, rng: np.random.Generator, p: float) -> tuple[np.ndarray, np.ndarray]:
    """T1 지시문 드롭아웃: 표본별로 확률 p로 지시문을 빈 지시(모든 토큰 PAD=0)로 바꾼 사본을 반환한다.

    빈 지시 정의: PAD만 있는 시퀀스. `encode_language`의 마스크 평균은 분모를 clamp_min(1)로 막으므로
    빈 지시의 언어 특징은 0 벡터 평균 → ReLU(lang_fc 편향)로 잘 정의된다(0으로 나누기 없음, 별도 null 임베딩 불필요).
    배포 때 지시문이 없으면 같은 PAD 시퀀스를 넣으면 학습 때의 "빈 지시" 경로와 같다.

    Args:
        tokens: int64 [B,L].
        rng: 전용 난수 생성기(다른 증강 난수와 섞지 않는다). p ≤ 0이면 난수를 소비하지 않는다.
        p: 드롭 확률 [0,1].

    Returns:
        (토큰 사본 또는 원본(p ≤ 0), 드롭 여부 bool [B]).
    """

    b = len(tokens)
    if p <= 0:
        return tokens, np.zeros(b, dtype=bool)
    drop = rng.random(b) < p
    out = np.array(tokens, dtype=np.int64, copy=True)
    out[drop] = 0
    return out, drop


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """확률 타깃 soft CE: mean_b(−Σ_c q_bc · log softmax(z_b)_c)."""

    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def add_feature_noise(features: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    """특징 증강: 가우시안 노이즈 N(0, σ²)를 더한 float32 사본을 반환한다(σ ≤ 0이면 그대로)."""

    if sigma <= 0:
        return features
    return (features + rng.normal(0.0, sigma, size=features.shape)).astype(np.float32)


def shift_images(image: np.ndarray, rng: np.random.Generator, max_shift: int = 2) -> np.ndarray:
    """샘플별 작은 평행이동(±max_shift px) 증강. image uint8 [B,H,W,C] → 같은 형태.

    가장자리 복제 패딩 후 잘라낸다. 두 시점(t, t−2) 채널이 함께 움직이므로 움직임 단서는 보존된다.
    uint8 상태에서 처리해 float 변환 전 비용을 줄인다(배치 128 기준 약 2 ms).
    """

    if max_shift <= 0:
        return image
    b, h, w = image.shape[:3]
    shifts = rng.integers(0, 2 * max_shift + 1, size=(b, 2))
    padded = np.pad(image, ((0, 0), (max_shift, max_shift), (max_shift, max_shift), (0, 0)), mode="edge")
    out = np.empty_like(image)
    for i, (dy, dx) in enumerate(shifts.tolist()):
        out[i] = padded[i, dy : dy + h, dx : dx + w]
    return out


def photometric_jitter(image: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """밝기 ±0.06·대비 ×[0.85, 1.15] 증강(image float [B,C,H,W], [0,1]).

    샘플마다 값을 하나 뽑아 두 시점 프레임에 똑같이 적용한다(시간 일관성 유지).
    """

    b = image.shape[0]
    contrast = torch.empty(b, 1, 1, 1).uniform_(0.85, 1.15, generator=gen).to(image.device)
    bright = torch.empty(b, 1, 1, 1).uniform_(-0.06, 0.06, generator=gen).to(image.device)
    mean = image.mean(dim=(1, 2, 3), keepdim=True)
    # (x − m)·c + m + b 를 제자리 연산으로 계산한다.
    return image.mul(contrast).add_(mean * (1.0 - contrast) + bright).clamp_(0.0, 1.0)


def augment_observation(
    obs: dict[str, np.ndarray], rng: np.random.Generator, max_shift: int = 2
) -> dict[str, np.ndarray]:
    """관측 dict의 영상에 평행이동 증강을 적용한 사본을 반환한다(밝기·대비는 텐서 변환 후 적용).

    좌우 반전은 하지 않는다(차선·주행 방향의 비대칭성을 깨뜨린다).
    """

    out = dict(obs)
    out["image"] = shift_images(obs["image"], rng, max_shift)
    return out


def _segment_means(values: list[float], n_bins: int) -> tuple[list[float], list[int]]:
    """손실 곡선을 n_bins 구간 평균으로 줄인다. 반환: (구간 평균, 구간 끝 단계)."""

    if not values:
        return [], []
    edges = np.linspace(0, len(values), min(n_bins, len(values)) + 1).astype(int)
    arr = np.asarray(values, dtype=np.float64)
    means = [float(arr[a:b].mean()) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    ends = [int(b) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    return means, ends


def _action_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    beta: float,
    loss_weight: np.ndarray | None,
    idx: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """청크 위치 가중 Huber 행동 손실.

    loss_weight가 None이면 v3와 같은 식(전체 평균)을 그대로 쓴다(비트 동일). 주어지면(v4 P4 위험 가중 손실)
    표본별 손실에 배치 안에서 평균 1로 정규화한 가중치를 곱해 평균한다.
    """

    per = F.smooth_l1_loss(pred, target, reduction="none", beta=beta) * weights
    if loss_weight is None:
        return per.mean()
    w = torch.from_numpy(loss_weight[idx]).to(device)
    w = w / w.mean()
    return (per.mean(1) * w).mean()


def train_policy(data: PolicyData, train_idx: np.ndarray, cfg: PolicyConfig) -> tuple[VLALitePolicy, dict[str, Any]]:
    """고정 경사 단계(cfg.steps) 동안 VLA-lite를 학습한다.

    - 미니배치: train_idx에서 복원 추출(numpy Generator(seed)). 데이터 양이 달라도 단계 수·배치는 같다.
    - 타깃: 같은 그룹 안 action[t:t+chunk](끝은 마지막 값으로 채움) / accel_scale(domain).
    - 손실: 청크 위치 가중 SmoothL1(Huber).
    - 최적화: AdamW, 선형 워밍업 + cosine 감쇠(→0), 기울기 노름 자르기.
    - 증강(cfg.augment): 평행이동 ±2px(numpy Generator(seed+1)), 밝기·대비(torch.Generator(seed+2)). 좌우 반전 없음.
      cfg.use_features이면 특징에 N(0, feature_noise²) 노이즈(numpy Generator(seed+3), 영상 증강 난수와 분리).
    - 특징(cfg.use_features): data.features [N, feature_dim]이 필요하다. False면 data.features가 있어도 쓰지 않는다.
      관측의 특징 이력은 cfg.feature_history·feature_stride 규칙(`PolicyData.observation(idx, history, stride)`)이다.
    - v4 P3(cfg.aux_weight > 0): data.aux_targets [N, aux_classes]가 필요하다. 손실 = 행동 손실 +
      aux_weight × soft CE(보조 로짓, aux_targets[t]).
    - v4 T1(cfg.lang_dropout > 0, use_language=True일 때): 표본별 확률 p로 지시문을 PAD만 있는 빈 지시로 바꾼다
      (numpy Generator(seed+4), 다른 난수와 분리).
    - 결정성: torch·numpy 시드 고정, 증강 난수는 전용 생성기를 쓴다. 스레드 수는 cfg.threads로 고정하고
      학습 후 원래 값으로 되돌린다. v4 필드가 기본값이면 v3와 같은 연산·난수 순서다.

    반환: (eval 모드 모델, 정보 dict: loss_curve, loss_curve_steps, final_loss, train_time_s, samples_per_s,
    n_params, vocab_size, n_train, config, final_aux_loss, aux_loss_curve, lang_dropped_frac).
    `loss_curve`·`final_loss`는 v3와 비교할 수 있도록 **행동 손실만**이다. `final_aux_loss`는 보조 soft CE
    (보조 헤드가 없으면 None), `lang_dropped_frac`는 빈 지시로 바꾼 표본 비율이다.
    """

    train_idx = np.asarray(train_idx, dtype=np.int64)
    if train_idx.size == 0:
        raise ValueError("train_idx가 비어 있습니다.")
    if cfg.steps <= 0 or cfg.batch <= 0:
        raise ValueError("steps와 batch는 양수여야 합니다.")
    if cfg.use_features:
        if data.features is None:
            raise ValueError("cfg.use_features=True인데 PolicyData.features가 없습니다.")
        if data.features.shape[1] != cfg.feature_dim:
            raise ValueError(f"PolicyData.features 차원({data.features.shape[1]})이 cfg.feature_dim({cfg.feature_dim})과 다릅니다.")
    use_aux = cfg.aux_weight > 0
    if use_aux:
        if data.aux_targets is None:
            raise ValueError("cfg.aux_weight > 0인데 PolicyData.aux_targets가 없습니다.")
        if data.aux_targets.shape[1] != cfg.aux_classes:
            raise ValueError(
                f"PolicyData.aux_targets 클래스 수({data.aux_targets.shape[1]})가 cfg.aux_classes({cfg.aux_classes})와 다릅니다."
            )
    use_lang_drop = cfg.lang_dropout > 0 and cfg.use_language

    prev_threads = torch.get_num_threads()
    torch.set_num_threads(max(1, int(cfg.threads)))
    try:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        aug_rng = np.random.default_rng(cfg.seed + 1)
        gen = torch.Generator().manual_seed(cfg.seed + 2)
        feat_rng = np.random.default_rng(cfg.seed + 3)
        lang_rng = np.random.default_rng(cfg.seed + 4)  # T1 전용(lang_dropout ≤ 0이면 소비하지 않는다)
        device = torch.device(cfg.device)

        vocab = _vocab_size(data.tokens)
        model = VLALitePolicy.from_config(cfg, vocab).to(device)
        model.train()
        n_params = count_parameters(model)
        a_scale = float(_obs().accel_scale(data.domain))
        weights = torch.from_numpy(chunk_weights(cfg.chunk, cfg.front_weight)).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        warmup = max(1, min(int(cfg.warmup_steps), cfg.steps // 20))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            prog = (step - warmup) / max(1, cfg.steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        logger.info(
            "VLA-lite 학습 시작: n_train=%d steps=%d batch=%d params=%d use_language=%s use_features=%s "
            "feature_encoder=%s H=%d s=%d aux_weight=%g lang_dropout=%g threads=%d",
            len(train_idx), cfg.steps, cfg.batch, n_params, cfg.use_language, cfg.use_features,
            cfg.feature_encoder, cfg.feature_history, cfg.feature_stride, cfg.aux_weight, cfg.lang_dropout, cfg.threads,
        )

        losses: list[float] = []
        aux_losses: list[float] = []
        n_dropped = 0
        t0 = time.perf_counter()
        for step in range(cfg.steps):
            idx = train_idx[rng.integers(0, len(train_idx), size=cfg.batch)]
            obs = data.observation(
                idx, history=cfg.feature_history, stride=cfg.feature_stride,
                proprio_history=cfg.proprio_history, proprio_stride=cfg.proprio_stride,
            )
            if cfg.augment:
                obs = augment_observation(obs, aug_rng)
                if cfg.use_features:
                    obs["features"] = add_feature_noise(obs["features"], feat_rng, cfg.feature_noise)
            if use_lang_drop:
                obs["tokens"], dropped = apply_lang_dropout(obs["tokens"], lang_rng, cfg.lang_dropout)
                n_dropped += int(dropped.sum())
            image, tokens, prop = _to_tensor_batch(obs, device)
            feats = _features_tensor(obs, model, device)
            if cfg.augment:
                image = photometric_jitter(image, gen)
            target = torch.from_numpy(data.action_chunk(idx, cfg.chunk) / a_scale).to(device)
            if use_aux:
                pred, aux_logits = model(image, tokens, prop, feats, return_aux=True)
                act_loss = _action_loss(pred, target, weights, cfg.huber_beta, data.loss_weight, idx, device)
                aux_loss = soft_cross_entropy(aux_logits, torch.from_numpy(data.aux_targets[idx]).to(device))
                loss = act_loss + cfg.aux_weight * aux_loss
            else:
                pred = model(image, tokens, prop, feats)
                loss = act_loss = _action_loss(pred, target, weights, cfg.huber_beta, data.loss_weight, idx, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            lv = float(act_loss.detach())
            if not math.isfinite(float(loss.detach())):
                raise FloatingPointError(f"손실이 유한하지 않습니다(step={step}).")
            losses.append(lv)
            if use_aux:
                aux_losses.append(float(aux_loss.detach()))
            if (step + 1) % max(1, cfg.steps // 10) == 0:
                logger.info("step %d/%d loss=%.4f", step + 1, cfg.steps, float(np.mean(losses[-50:])))
        elapsed = time.perf_counter() - t0
    finally:
        torch.set_num_threads(prev_threads)

    model.eval()
    curve, curve_steps = _segment_means(losses, cfg.n_curve_bins)
    tail = max(1, cfg.steps // 20)
    info: dict[str, Any] = {
        "loss_curve": curve,
        "loss_curve_steps": curve_steps,
        "final_loss": float(np.mean(losses[-tail:])),
        "train_time_s": float(elapsed),
        "samples_per_s": float(cfg.steps * cfg.batch / max(elapsed, 1e-9)),
        "n_params": n_params,
        "vocab_size": vocab,
        "n_train": int(len(train_idx)),
        "config": asdict(cfg),
        # v4: 보조 손실(행동 손실과 분리 기록, 보조 헤드가 없으면 None)·지시문 드롭 비율
        "final_aux_loss": float(np.mean(aux_losses[-tail:])) if aux_losses else None,
        "aux_loss_curve": _segment_means(aux_losses, cfg.n_curve_bins)[0] if aux_losses else [],
        "lang_dropped_frac": float(n_dropped / (cfg.steps * cfg.batch)),
    }
    logger.info("VLA-lite 학습 완료: %.1fs, %.0f samples/s, final_loss=%.4f", elapsed, info["samples_per_s"], info["final_loss"])
    return model, info


# ---------------------------------------------------------------------------
# 추론
# ---------------------------------------------------------------------------
def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def make_policy_fn(model: VLALitePolicy, domain: str) -> PolicyFn:
    """폐루프용 배치 정책 함수를 만든다(계약 `PolicyFn`).

    입력: {"image": uint8 [B,H,W,6], "tokens": int64 [B,L], "proprio": float32 [B,1](obs.proprio 정규화 값),
          "features": float32 [B,feature_history,feature_dim](v3·v4, 모델이 use_features일 때만 필수, 아니면 무시)}
    출력: float32 [B] 물리 단위 가속도 명령 = 청크 첫 원소 × accel_scale(domain). eval 모드, no_grad.
    보조 헤드는 계산하지 않는다.

    반환 함수 속성(v4, 폐루프가 특징 이력 규칙을 읽는다): `policy_fn.feature_history`, `policy_fn.feature_stride`,
    `policy_fn.use_features`. 폐루프(`FeatureHistory`)는 이 규칙으로 [B,H,D]를 만들어 그대로 넘긴다.

    Raises(호출 시):
        ValueError: use_features=True 모델인데 입력에 "features"가 없거나 형태가 [B,H,D]가 아닐 때.
    """

    a_scale = float(_obs().accel_scale(domain))

    def policy_fn(batch: dict[str, np.ndarray]) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            device = _model_device(model)
            feats = _features_tensor(batch, model, device)
            image, tokens, prop = _to_tensor_batch(batch, device)
            out = model(image, tokens, prop, feats)[:, 0] * a_scale
        return out.cpu().numpy().astype(np.float32)

    fn: Any = policy_fn
    fn.feature_history = int(model.feature_history)
    fn.feature_stride = int(model.feature_stride)
    fn.use_features = bool(model.use_features)
    fn.proprio_history = int(model.proprio_history)  # P5: 폐루프가 "proprio" [B,Hp] 이력을 만든다
    fn.proprio_stride = int(model.proprio_stride)
    return policy_fn


def predict_open_loop(model: VLALitePolicy, data: PolicyData, idx: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """개루프 예측: float32 [len(idx), chunk] 물리 단위 가속도(m/s²).

    특징 이력은 모델의 `feature_history`·`feature_stride`(학습 설정과 같은 값) 규칙으로 만든다.
    """

    idx = np.asarray(idx, dtype=np.int64)
    if model.use_features and data.features is None:
        raise ValueError("use_features=True 모델의 개루프 예측에는 PolicyData.features가 필요합니다.")
    a_scale = float(_obs().accel_scale(data.domain))
    device = _model_device(model)
    outs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            obs = data.observation(
                idx[s : s + batch_size], history=model.feature_history, stride=model.feature_stride,
                proprio_history=model.proprio_history, proprio_stride=model.proprio_stride,
            )
            image, tokens, prop = _to_tensor_batch(obs, device)
            outs.append((model(image, tokens, prop, _features_tensor(obs, model, device)) * a_scale).cpu().numpy())
    if not outs:
        return np.zeros((0, model.chunk), dtype=np.float32)
    return np.concatenate(outs, axis=0).astype(np.float32)


__all__ = [
    "PolicyConfig",
    "VLALitePolicy",
    "ImageSource",
    "ArrayImageSource",
    "PoolImageSource",
    "PolicyData",
    "PolicyFn",
    "chunk_weights",
    "shift_images",
    "photometric_jitter",
    "augment_observation",
    "add_feature_noise",
    "apply_lang_dropout",
    "soft_cross_entropy",
    "FEATURE_ENCODERS",
    "count_parameters",
    "train_policy",
    "make_policy_fn",
    "predict_open_loop",
]
